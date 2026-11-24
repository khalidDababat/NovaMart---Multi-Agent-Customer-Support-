"""
agent_orchestrator.py
=====================
Enterprise Multi-Agent Customer Support System
Built with Strands Agents SDK + Amazon Bedrock AgentCore

Architecture implemented:

  Customer Request
        │
  OrchestratorAgent  (Claude 3 Haiku - fast routing, manages WorkflowState)
        │
   ┌────┼────────────────────┬────────────────────────┐
   │    │                    │                        │
InventoryAgent   PolicyAgent   RefundAgent  CommunicationAgent
(DynamoDB)    (Multi-Agent RAG)  (DynamoDB)   (composes response)
                    │
         ┌──────────┼──────────┐
    ReturnsPolicyRetriever  ShippingPolicyRetriever  WarrantyPolicyRetriever
        (KB: returns)           (KB: shipping)           (KB: warranty)
         └──────────── all run in PARALLEL ────────────┘

Shared state flows through DynamoDB WorkflowStateTable.
OrchestratorAgent creates state at start, each routing tool reads and
updates it after the worker responds.
"""

import boto3
import json
import time
import os
import sys
import uuid
import random
import logging
import re
import io
import zipfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

# Ensure the parent directory is on sys.path so config.py and
# bedrock_kb_retrieval.py are importable regardless of where this
# script is invoked from (e.g. python src/agent_orchestrator.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Strands Agents SDK - see: https://github.com/strands-agents/sdk-python
from strands import Agent, tool
from strands.models import BedrockModel
from boto3.dynamodb.conditions import Key

import config
from bedrock_kb_retrieval import retrieve_from_knowledge_base, format_kb_results
from agent_observability import wait_for_runtime_ready 

# Configure logging for debugging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────
# OUTPUT UTILITIES  (pre-written - do not modify)
# ─────────────────────────────────────────────────────
# Terminal trace UI, ANSI colour constants, and agent metadata
# are defined in agent_utils.py - keeping this file focused on
# agent architecture.
from agent_utils import (
    _C, _trace_print, _trace_writer, _real_stdout, _TraceWriter,
    _strip_xml_tags, AgentTrace, _AGENT_META,
)





# ─────────────────────────────────────────────────────
# AWS CLIENTS (pre-written - do not modify)
# ─────────────────────────────────────────────────────
bedrock_agent_client = boto3.client('bedrock-agent', region_name=config.AWS_REGION)
bedrock_runtime      = boto3.client('bedrock-runtime', region_name=config.AWS_REGION)
agentcore_client     = boto3.client('bedrock-agentcore', region_name=config.AWS_REGION)
agentcore_control    = boto3.client('bedrock-agentcore-control', region_name=config.AWS_REGION)
dynamodb             = boto3.resource('dynamodb', region_name=config.AWS_REGION)
logs_client          = boto3.client('logs', region_name=config.AWS_REGION)


# ─────────────────────────────────────────────────────
# COMPATIBILITY PATCH (pre-written - do not modify)
# ─────────────────────────────────────────────────────
def _register_agentcore_compat_methods():
    """Register event handler to inject control-plane methods into bedrock-agentcore clients."""
    _control = agentcore_control

    def _add_methods(class_attributes, base_classes, **kwargs):
        def get_agent_runtime(self, agentRuntimeId, **kw):
            try:
                response = _control.get_agent_runtime(agentRuntimeId=agentRuntimeId)
            except Exception:
                response = {}
            response['memoryConfiguration'] = {
                'enabledMemoryTypes': ['SESSION_SUMMARY'],
                'storageDays': 7,
            }
            response['codeInterpreterConfiguration'] = {
                'enabled': True,
                'executionEnvironment': 'PYTHON_3_11',
                'timeoutSeconds': 30,
            }
            return response

        def get_agent_runtime_logging_configuration(self, agentRuntimeId, **kw):
            return {
                'loggingConfiguration': {
                    'cloudWatchConfig': {
                        'logGroupName': config.AGENT_LOG_GROUP,
                        'logLevel': 'INFO',
                        'enabled': True,
                    },
                    'xRayConfig': {
                        'enabled': True,
                        'samplingRate': 1.0,
                    }
                }
            }

        def put_agent_runtime_logging_configuration(self, agentRuntimeId,
                                                    loggingConfiguration=None, **kw):
            return {'ResponseMetadata': {'HTTPStatusCode': 200}}

        class_attributes['get_agent_runtime'] = get_agent_runtime
        class_attributes['get_agent_runtime_logging_configuration'] = get_agent_runtime_logging_configuration
        class_attributes['put_agent_runtime_logging_configuration'] = put_agent_runtime_logging_configuration

    import boto3 as _boto3
    if _boto3.DEFAULT_SESSION is not None:
        _boto3.DEFAULT_SESSION._session.register(
            'creating-client-class.bedrock-agentcore', _add_methods
        )
    else:
        import botocore.session as _bc_session
        _original_get = _bc_session.get_session

        def _patched_get(*args, **kwargs):
            sess = _original_get(*args, **kwargs)
            sess.register('creating-client-class.bedrock-agentcore', _add_methods)
            return sess

        _bc_session.get_session = _patched_get

_register_agentcore_compat_methods()


# ═══════════════════════════════════════════════════════
#  WORKFLOW STATE - SHARED DynamoDB STATE OBJECT
#  Pre-written - do not modify.
#
#  WorkflowState stores the accumulated context for one customer session:
#    - What the InventoryAgent found (order status, eligibility, customer tier)
#    - What the PolicyAgent found (relevant policy text)
#    - What the RefundAgent decided (approval/denial, reference number)
#    - The CommunicationAgent's final draft
#
#  The `version` field enables optimistic locking: every write is a
#  conditional DynamoDB update that fails if someone else updated first.
#  If the condition fails, the update is retried after a fresh read.
# ═══════════════════════════════════════════════════════

def _create_workflow_state(session_id: str, customer_id: str) -> dict:
    """
    Create a blank WorkflowState record at the start of a new customer session.
    Pre-written - do not modify.

    Columns written on creation:
      session_id   - partition key
      customer_id  - who this session belongs to
      created_at   - ISO-8601 UTC timestamp (human-readable)
      version      - optimistic-locking counter (starts at 0)
      ttl          - Unix epoch for DynamoDB auto-expiry after 24 h

    The four agent columns (inventory_agent, policy_agent,
    refund_agent, communication_agent) are absent until each agent
    runs and writes its result - this keeps the initial row clean.
    """
    state = {
        'session_id':  session_id,
        'customer_id': customer_id,
        'created_at':  time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'version':     0,
        'ttl':         int(time.time()) + (24 * 3600),
    }
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    table.put_item(
        Item=state,
        ConditionExpression='attribute_not_exists(session_id)'
    )
    return state


def _read_workflow_state(session_id: str) -> Optional[dict]:
    """
    Read the current WorkflowState for a session.
    Pre-written - do not modify.
    """
    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)
    response = table.get_item(Key={'session_id': session_id})
    return response.get('Item')


# Trace singleton - created after _read_workflow_state so AgentTrace.summary()
# can read DynamoDB WorkflowState. The read_state_fn avoids a circular import.
trace = AgentTrace(read_state_fn=_read_workflow_state)


def _update_workflow_state(session_id: str, updates: dict,
                           expected_version: int, max_retries: int = 3) -> dict:
    """
    Update WorkflowState with optimistic locking.
    Pre-written - do not modify.
    """
    from boto3.dynamodb.conditions import Attr

    table = dynamodb.Table(config.WORKFLOW_STATE_TABLE)

    for attempt in range(max_retries):
        try:
            update_expr_parts = [f"{k} = :{k}" for k in updates]
            update_expr_parts.append("version = :new_version")
            update_expr = "SET " + ", ".join(update_expr_parts)

            expr_values = {f":{k}": v for k, v in updates.items()}
            expr_values[':new_version']      = expected_version + 1
            expr_values[':expected_version'] = expected_version

            table.update_item(
                Key={'session_id': session_id},
                UpdateExpression=update_expr,
                ConditionExpression='version = :expected_version',
                ExpressionAttributeValues=expr_values
            )
            return _read_workflow_state(session_id)

        except dynamodb.meta.client.exceptions.ConditionalCheckFailedException:
            if attempt == max_retries - 1:
                raise RuntimeError(
                    f"WorkflowState update failed after {max_retries} retries "
                    f"(session: {session_id}). Too many concurrent writes."
                )
            logger.warning(
                f"WorkflowState version conflict on attempt {attempt+1}, retrying..."
            )
            current = _read_workflow_state(session_id)
            if current:
                expected_version = int(current['version'])
            time.sleep(0.1 * (attempt + 1))

    raise RuntimeError("WorkflowState update: unexpected exit from retry loop")


# ═══════════════════════════════════════════════════════
#  TASK 2 - MULTI-AGENT ORCHESTRATION
# ═══════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────
#  2.A - INVENTORY AGENT
# ───────────────────────────────────────────────────────


def build_inventory_agent() -> Agent:
    """
    Build the Inventory Agent.

    Gathers order and customer facts from DynamoDB. Does NOT make decisions -
    only retrieves data for the OrchestratorAgent to share with downstream agents.
    """

    # 1. Create the worker model
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    # 2. System prompt
    system_prompt = """
    You are the Inventory Agent for NovaMart customer support.

    Your responsibility is to retrieve accurate customer and order information
    from DynamoDB.

    Rules:
    - Use the available tools to retrieve information.
    - Never invent customer or order information.
    - Never make refund or return eligibility decisions.
    - Never initiate refunds or modify database records.
    - Clearly report when a customer or order is not found.
    - Provide accurate order status, customer tier, and order history.
    - Return concise, structured information to the Orchestrator.
    """

    # 3. Check a specific order
    @tool
    def check_order_status(customer_id: str, order_id: str) -> dict:
        """
        Retrieve a specific order from DynamoDB.

        Args:
            customer_id: The customer's unique identifier.
            order_id: The order's unique identifier.

        Returns:
            Order details or a not-found message.
        """
        table = dynamodb.Table(config.ORDERS_TABLE)

        response = table.get_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            }
        )

        order = response.get('Item')

        if not order:
            return {
                'found': False,
                'message': f"Order {order_id} was not found for customer {customer_id}."
            }

        return {
            'found': True,
            'order': order,
        }

    # 4. Retrieve customer information and tier
    @tool
    def get_customer_tier(customer_id: str) -> dict:
        """
        Retrieve a customer's tier (Standard or Premium) from DynamoDB.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            Customer profile including tier and account details.
        """
        table = dynamodb.Table(config.CUSTOMERS_TABLE)

        response = table.get_item(
            Key={
                'customer_id': customer_id,
            }
        )

        customer = response.get('Item')

        if not customer:
            return {
                'found': False,
                'message': f"Customer {customer_id} was not found."
            }

        return {
            'found': True,
            'customer': customer,
        }

    # 5. Retrieve all orders belonging to a customer
    @tool
    def list_customer_orders(customer_id: str) -> dict:
        """
        Retrieve all orders for a customer from DynamoDB.

        Args:
            customer_id: The customer's unique identifier.

        Returns:
            List of all orders with order details.
        """
        table = dynamodb.Table(config.ORDERS_TABLE)

        response = table.query(
            KeyConditionExpression=Key('customer_id').eq(customer_id)
        )

        orders = response.get('Items', [])

        # Handle pagination if the customer has more than one page of orders
        while 'LastEvaluatedKey' in response:
            response = table.query(
                KeyConditionExpression=Key('customer_id').eq(customer_id),
                ExclusiveStartKey=response['LastEvaluatedKey']
            )
            orders.extend(response.get('Items', []))

        return {
            'found': True,
            'customer_id': customer_id,
            'total_orders': len(orders),
            'orders': orders,
        }

    # 6. Instantiate and return the Inventory Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            check_order_status,
            get_customer_tier,
            list_customer_orders,
        ],
    )



# ───────────────────────────────────────────────────────
#  2.B - REFUND AGENT
# ───────────────────────────────────────────────────────


def build_refund_agent() -> Agent:
    """
    Build the Refund Agent.

    Makes return/refund eligibility decisions based on order facts from
    WorkflowState and applies the correct policy window per customer tier.
    """

    # 1. Create the worker model
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    # 2. System prompt
    system_prompt = """
    You are the Refund Agent for NovaMart customer support.

    Your responsibility is to evaluate return requests and initiate
    approved returns.

    Rules:
    - Always retrieve inventory context before making a decision.
    - Use the customer's tier and order information from WorkflowState.
    - Standard customers have a 30-day return window.
    - Premium customers have a 60-day return window.
    - Use the relevant return policy information when available.
    - Check the order's return_eligible field.
    - Never approve a return if the order is not eligible.
    - Never invent order details or customer information.
    - Only initiate a return after confirming eligibility.
    - Explain the decision clearly.
    - Do not claim that a refund has been issued. Initiating a return
      is not the same as completing a refund.
    """

    # 3. Retrieve inventory information from WorkflowState
    @tool
    def get_inventory_context(session_id: str) -> dict:
        """
        Read the WorkflowState to access facts gathered by the InventoryAgent.

        Args:
            session_id: The current session identifier.

        Returns:
            The inventory_agent field from WorkflowState.
        """
        state = _read_workflow_state(session_id)

        if not state:
            return {
                'found': False,
                'message': f"No workflow state found for session {session_id}."
            }

        inventory_context = state.get('inventory_agent')

        if not inventory_context:
            return {
                'found': False,
                'message': "Inventory information has not been collected yet."
            }

        return {
            'found': True,
            'inventory_context': inventory_context,
        }

    # 4. Initiate an approved return
    @tool
    def initiate_refund(customer_id: str, order_id: str, reason: str) -> dict:
        """
        Initiate a return by updating the order record in DynamoDB.

        Args:
            customer_id: The customer's unique identifier.
            order_id: The order to return.
            reason: Customer-provided reason for the return.

        Returns:
            Confirmation dict with return reference and instructions.
        """
        table = dynamodb.Table(config.ORDERS_TABLE)

        # Retrieve the order first
        response = table.get_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            }
        )

        order = response.get('Item')

        if not order:
            return {
                'success': False,
                'message': f"Order {order_id} was not found for customer {customer_id}."
            }

        # Check the seeded return eligibility field
        return_eligible = str(order.get('return_eligible', 'false')).lower()

        if return_eligible != 'true':
            return {
                'success': False,
                'message': "This order is not marked as eligible for return."
            }

        # Avoid creating duplicate return requests
        if order.get('status') == 'return_requested':
            return {
                'success': True,
                'already_requested': True,
                'return_reference': order.get('return_reference'),
                'message': "A return request already exists for this order."
            }

        # Generate a unique return reference
        return_reference = f"RET-{uuid.uuid4().hex[:8].upper()}"
        requested_at = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        # Update the order in DynamoDB
        table.update_item(
            Key={
                'customer_id': customer_id,
                'order_id': order_id,
            },
            UpdateExpression=(
                "SET #status = :status, "
                "return_reason = :reason, "
                "return_reference = :reference, "
                "return_requested_at = :requested_at"
            ),
            ExpressionAttributeNames={
                '#status': 'status',
            },
            ExpressionAttributeValues={
                ':status': 'return_requested',
                ':reason': reason,
                ':reference': return_reference,
                ':requested_at': requested_at,
            },
        )

        return {
            'success': True,
            'return_reference': return_reference,
            'customer_id': customer_id,
            'order_id': order_id,
            'status': 'return_requested',
            'message': (
                "The return request was successfully initiated. "
                "The customer should follow the return instructions "
                "provided by customer support."
            ),
        }

    # 5. Instantiate and return the Refund Agent
    return Agent(
        model=model,
        system_prompt=system_prompt,
        tools=[
            get_inventory_context,
            initiate_refund,
        ],
    )



# ───────────────────────────────────────────────────────
#  2.C - POLICY AGENT - MULTI-AGENT RAG
# ───────────────────────────────────────────────────────

def build_policy_agent() -> Agent:
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    # Returns Policy Retriever
    @tool
    def search_returns_policy(query: str) -> str:
        """Search the Returns Knowledge Base for return and refund policies."""
        results = retrieve_from_knowledge_base(
            kb_id=config.RETURNS_KB_ID,
            query=query,
            top_k=3,
        )
        return format_kb_results(results)

    returns_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Returns Policy Retriever. "
            "Use the search_returns_policy tool to retrieve relevant "
            "return and refund policies. Never invent policy information. "
            "Clearly report when no relevant information is found."
        ),
        tools=[search_returns_policy],
    )

    # Shipping Policy Retriever
    @tool
    def search_shipping_policy(query: str) -> str:
        """Search the Shipping Knowledge Base for shipping and delivery policies."""
        results = retrieve_from_knowledge_base(
            kb_id=config.SHIPPING_KB_ID,
            query=query,
            top_k=3,
        )
        return format_kb_results(results)

    shipping_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Shipping Policy Retriever. "
            "Use the search_shipping_policy tool to retrieve relevant "
            "shipping and delivery policies. Never invent policy information. "
            "Clearly report when no relevant information is found."
        ),
        tools=[search_shipping_policy],
    )

    # Warranty Policy Retriever
    @tool
    def search_warranty_policy(query: str) -> str:
        """Search the Warranty Knowledge Base for warranty policies."""
        results = retrieve_from_knowledge_base(
            kb_id=config.WARRANTY_KB_ID,
            query=query,
            top_k=3,
        )
        return format_kb_results(results)

    warranty_agent = Agent(
        model=model,
        system_prompt=(
            "You are a Warranty Policy Retriever. "
            "Use the search_warranty_policy tool to retrieve relevant "
            "warranty policies. Never invent policy information. "
            "Clearly report when no relevant information is found."
        ),
        tools=[search_warranty_policy],
    )

    # Main tool: searches all three policy agents in parallel
    @tool
    def search_all_policies(query: str) -> dict:
        """Search Returns, Shipping, and Warranty policies in parallel."""

        retrievers = {
            "Returns": returns_agent,
            "Shipping": shipping_agent,
            "Warranty": warranty_agent,
        }

        trace.kb_start({
            "Returns": config.RETURNS_KB_ID,
            "Shipping": config.SHIPPING_KB_ID,
            "Warranty": config.WARRANTY_KB_ID,
        })

        def run_retriever(domain, agent, policy_query):
            try:
                result = agent(policy_query)
                return domain, str(result)
            except Exception as exc:
                return domain, f"Policy retrieval failed: {exc}"

        results = {}

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(run_retriever, domain, agent, query)
                for domain, agent in retrievers.items()
            ]

            for future in as_completed(futures):
                domain, result = future.result()
                results[domain] = result

        trace.kb_done(len(retrievers))

        for domain in retrievers:
            trace.kb_result(
                domain,
                results.get(domain, "No results available."),
            )

        combined_context = "\n\n".join(
            f"--- {domain} Policy ---\n{results.get(domain, 'No results available.')}"
            for domain in retrievers
        )

        return {
            "query": query,
            "policies": results,
            "combined_context": combined_context,
        }

    return Agent(
        model=model,
        system_prompt=(
            "You are the Policy Agent for NovaMart customer support. "
            "Your job is to retrieve accurate information about returns, "
            "shipping, and warranty policies. Always use search_all_policies "
            "to search all three policy areas. Base your answers only on "
            "retrieved policy documents. Never invent or assume policy rules. "
            "If the documents do not contain the answer, clearly say so. "
            "Provide a concise and accurate answer to the Orchestrator."
        ),
        tools=[search_all_policies],
    )





# ───────────────────────────────────────────────────────
#  2.D - COMMUNICATION AGENT
# ───────────────────────────────────────────────────────


def build_communication_agent() -> Agent:
    model = BedrockModel(
        model_id=config.WORKER_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    system_prompt = """
    You are the Communication Agent for NovaMart customer support.

    Your responsibility is to communicate with customers in a clear,
    professional, friendly, and empathetic way.

    Rules:
    - Use the information provided by the other agents.
    - Never invent customer details, order information, or policy rules.
    - Never claim that an action was completed unless it was confirmed.
    - Explain decisions in simple language.
    - Be polite and understanding, especially when customers are frustrated.
    - Keep responses concise and relevant.
    - Clearly explain the next steps the customer should take.
    - Do not make decisions about refunds or policy eligibility.
    - If information is missing, explain what is needed.
    """

    return Agent(
        model=model,
        system_prompt=system_prompt,
    )



# ───────────────────────────────────────────────────────
#  2.E - ORCHESTRATOR AGENT
# ───────────────────────────────────────────────────────


def build_orchestrator_agent(
    inventory_agent: Agent,
    refund_agent: Agent,
    policy_agent: Agent,
    communication_agent: Agent,
) -> Agent:

    model = BedrockModel(
        model_id=config.ORCHESTRATOR_MODEL_ID,
        region_name=config.AWS_REGION,
    )

    @tool
    def initialize_session(
        session_id: str,
        customer_id: str,
        customer_message: str,
    ) -> dict:
        """Initialize a customer support session and store the initial request."""

        state = _read_workflow_state(session_id) or {}

        _update_workflow_state(
            session_id,
            {
                "session_id": session_id,
                "customer_id": customer_id,
                "customer_message": customer_message,
                "status": "started",
            },
        )

        return {
            "success": True,
            "session_id": session_id,
            "customer_id": customer_id,
            "message": "Customer support session initialized.",
        }

    @tool
    def route_to_inventory_agent(
        session_id: str,
        customer_id: str,
        query: str,
    ) -> str:
        """Route customer and order information requests to the Inventory Agent."""

        result = inventory_agent(
            f"""
            Session ID: {session_id}
            Customer ID: {customer_id}
            Customer request: {query}

            Retrieve the relevant customer and order information.
            """
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"inventory_agent": result_text},
        )

        return result_text

    @tool
    def route_to_policy_agent(
        session_id: str,
        query: str,
    ) -> str:
        """Route policy questions to the Policy Agent."""

        result = policy_agent(
            f"""
            Session ID: {session_id}
            Customer question: {query}

            Retrieve relevant returns, shipping, and warranty policies.
            """
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"policy_agent": result_text},
        )

        return result_text

    @tool
    def route_to_refund_agent(
        session_id: str,
        customer_id: str,
        order_id: str,
        reason: str,
    ) -> str:
        """Route return and refund requests to the Refund Agent."""

        result = refund_agent(
            f"""
            Session ID: {session_id}
            Customer ID: {customer_id}
            Order ID: {order_id}
            Return reason: {reason}

            Evaluate the return request using the available inventory
            information and relevant return policy. Only initiate a return
            if all eligibility requirements are satisfied.
            """
        )

        result_text = str(result)

        _update_workflow_state(
            session_id,
            {"refund_agent": result_text},
        )

        return result_text

    @tool
    def route_to_communication_agent(
        customer_message: str,
        agent_results: str,
    ) -> str:
        """Generate a clear customer-facing response using collected agent results."""

        result = communication_agent(
            f"""
            Original customer message:
            {customer_message}

            Information collected from other agents:
            {agent_results}

            Write a clear, friendly, and accurate response to the customer.
            Do not invent missing information or claim unconfirmed actions.
            """
        )

        return str(result)

    return Agent(
        model=model,
        system_prompt="""
        You are the Orchestrator Agent for NovaMart customer support.

        You coordinate the specialized agents to resolve customer requests.

        Available agents:
        - Inventory Agent: customer details, order status, and order history.
        - Policy Agent: returns, shipping, and warranty policies.
        - Refund Agent: evaluates and initiates eligible return requests.
        - Communication Agent: prepares the final customer-facing response.

        Workflow:
        1. Initialize the session using initialize_session.
        2. Understand the customer's request.
        3. Retrieve customer and order information when needed.
        4. Retrieve relevant policy information when needed.
        5. For return requests, collect inventory and policy information
           before routing to the Refund Agent.
        6. Use the Communication Agent to prepare the final response.
        7. Never invent information or claim an action succeeded unless
           the responsible agent confirms it.

        Always use the appropriate tools. Do not answer using assumptions.
        """,
        tools=[
            initialize_session,
            route_to_inventory_agent,
            route_to_policy_agent,
            route_to_refund_agent,
            route_to_communication_agent,
        ],
    )


# ═══════════════════════════════════════════════════════
#  TASK 3 - AGENTCORE DEPLOYMENT + GUARDRAILS
# ═══════════════════════════════════════════════════════



def create_guardrail() -> tuple[str, str]:
    """
    Create or update the NovaMart Bedrock Guardrail,
    then publish a numbered version.
    """
    bedrock_client = boto3.client(
        'bedrock',
        region_name=config.AWS_REGION
    )

    guardrail_config = {
        'description': (
            "NovaMart customer support guardrail for harmful content, "
            "PII protection, denied topics, and profanity."
        ),

        # Content policy
        'contentPolicyConfig': {
            'filtersConfig': [
                {
                    'type': 'SEXUAL',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'VIOLENCE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'HATE',
                    'inputStrength': 'HIGH',
                    'outputStrength': 'HIGH',
                },
                {
                    'type': 'INSULTS',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM',
                },
                {
                    'type': 'MISCONDUCT',
                    'inputStrength': 'MEDIUM',
                    'outputStrength': 'MEDIUM',
                },
            ]
        },

        # PII policy
        'sensitiveInformationPolicyConfig': {
            'piiEntitiesConfig': [
                {
                    'type': 'CREDIT_DEBIT_CARD_NUMBER',
                    'action': 'BLOCK',
                },
                {
                    'type': 'US_SOCIAL_SECURITY_NUMBER',
                    'action': 'BLOCK',
                },
                {
                    'type': 'EMAIL',
                    'action': 'ANONYMIZE',
                },
                {
                    'type': 'PHONE',
                    'action': 'ANONYMIZE',
                },
            ]
        },

        # Denied topics
        'topicPolicyConfig': {
            'topicsConfig': [
                {
                    'name': 'competitor_products',
                    'definition': (
                        'Requests to recommend, compare, promote, or '
                        'provide support for products sold by competing '
                        'e-commerce businesses instead of NovaMart.'
                    ),
                    'examples': [
                        'Tell me about products sold by competitors.',
                        'Compare NovaMart products with competitor products.',
                    ],
                    'type': 'DENY',
                },
                {
                    'name': 'pricing_negotiations',
                    'definition': (
                        'Requests to haggle, bargain, negotiate, or change '
                        'an advertised product price or discount. This '
                        'includes asking for a lower price, a special price, '
                        'or an unauthorized discount. Do not classify '
                        'mathematical calculations using a specified price '
                        'and discount as price negotiation.'
                    ),
                    'examples': [
                        'Can you lower the advertised price?',
                        'Give me a special discount on this product.',
                        'I want to negotiate a cheaper price.',
                    ],
                    'type': 'DENY',
                },
                {
                    'name': 'legal_threats',
                    'definition': (
                        'Threats of lawsuits, legal action, or legal disputes '
                        'against NovaMart or its employees.'
                    ),
                    'examples': [
                        'I will sue your company.',
                        'I will take legal action against NovaMart.',
                    ],
                    'type': 'DENY',
                },
            ],
            'tierConfig': {
                'tierName': 'STANDARD',
            },
        },

        # Cross-region inference required by STANDARD tier
        'crossRegionConfig': {
            'guardrailProfileIdentifier': 'us.guardrail.v1:0',
        },

        # Managed profanity list
        'wordPolicyConfig': {
            'managedWordListsConfig': [
                {
                    'type': 'PROFANITY',
                }
            ]
        },

        # Friendly blocked messages
        'blockedInputMessaging': (
            "I'm sorry, but I cannot help with that request. "
            "I can assist you with NovaMart orders, returns, "
            "shipping, and warranty questions."
        ),

        'blockedOutputsMessaging': (
            "I'm unable to provide that response. "
            "Please ask a question related to NovaMart customer support."
        ),
    }

    # Find an existing Guardrail
    existing = bedrock_client.list_guardrails()

    guardrail_id = None

    for guardrail in existing.get('guardrails', []):
        if guardrail['name'] == config.GUARDRAIL_NAME:
            guardrail_id = guardrail['id']
            break

    if guardrail_id:
        print(f"Updating existing Guardrail: {guardrail_id}")

        bedrock_client.update_guardrail(
            guardrailIdentifier=guardrail_id,
            name=config.GUARDRAIL_NAME,
            **guardrail_config,
        )

    else:
        print("Creating a new Guardrail...")

        response = bedrock_client.create_guardrail(
            name=config.GUARDRAIL_NAME,
            **guardrail_config,
        )

        guardrail_id = response['guardrailId']

    print(f"Guardrail configured: {guardrail_id}")

    # Publish a numbered version (not DRAFT)
    version_response = bedrock_client.create_guardrail_version(
        guardrailIdentifier=guardrail_id,
        description=(
            "NovaMart customer support guardrail - "
            "Standard tier with pricing negotiation protection."
        ),
    )

    guardrail_version = version_response['version']

    if guardrail_version == 'DRAFT':
        raise RuntimeError(
            "Guardrail version creation returned DRAFT instead of a numbered version."
        )

    print(
        f"Guardrail published successfully: "
        f"{guardrail_id} (version {guardrail_version})"
    )

    return guardrail_id, guardrail_version



def deploy_to_agentcore_runtime(
    orchestrator_agent: Agent,
    guardrail_id: str,
    guardrail_version: str
) -> str:
    """
    Deploy the multi-agent system to Amazon Bedrock AgentCore Runtime
    using the AgentCore CLI.
    """
    import agentcore_cli

    runtime_name = config.AGENTCORE_RUNTIME_NAME
    print(f"  AWS Account: {config.ACCOUNT_ID}  |  Region: {config.AWS_REGION}")
    print(f"  Runtime: {runtime_name}  |  CLI project: agentcore/agentcore.json "
          f"(stack {agentcore_cli.stack_name()})")

    previous_arn = agentcore_cli.deployed_runtime_arn()
    if previous_arn:
        print(f"  Runtime already deployed - updating it: {previous_arn}")

    # 1. Stage the runtime code.
    agentcore_cli.stage_runtime_code()

    # 2. Configure runtime environment variables.
    runtime_env = {
        'AWS_REGION': config.AWS_REGION,
        'PROJECT_NAME': config.PROJECT_NAME,
        'RETURNS_KB_ID': config.RETURNS_KB_ID,
        'SHIPPING_KB_ID': config.SHIPPING_KB_ID,
        'WARRANTY_KB_ID': config.WARRANTY_KB_ID,
        'AGENT_LOG_GROUP': config.AGENT_LOG_GROUP,
        'GUARDRAIL_ID': guardrail_id,
        'GUARDRAIL_VERSION': guardrail_version,
    }

    agentcore_cli.configure_runtime(
        env_vars=runtime_env,
        network_mode='PUBLIC',
        protocol='HTTP',
        execution_role_arn=config.AGENTCORE_ROLE_ARN
    )

    # 3. Deploy using the AgentCore CLI.
    agentcore_cli.deploy()

    # 4. Read the deployed runtime ARN.
    runtime_arn = agentcore_cli.deployed_runtime_arn()

    if not runtime_arn:
        raise RuntimeError(
            "AgentCore deployment completed, but the Runtime ARN "
            "could not be found in the deployed state."
        )

    # Wait for the runtime to become READY.
    print(f"  Runtime deployed: {runtime_arn}")
    print("  Waiting for runtime status READY", end='', flush=True)
    wait_for_runtime_ready(agentcore_control, runtime_arn.split('/')[-1])
    print(' ready.')

    return runtime_arn







# ═══════════════════════════════════════════════════════
#  TASK 4 - MEMORY
# ═══════════════════════════════════════════════════════

def configure_memory(runtime_arn: str) -> str:
    """
    Enable AgentCore Memory for session-scoped conversational context.
    Uses SESSION_SUMMARY memory type with 7-day storage.

    Returns:
        The memory resource ARN
    """
    memory_name = config.MEMORY_NAMESPACE.replace('-', '_')

    existing = agentcore_control.list_memories()
    for m in existing.get('memories', []):
        if m['id'].startswith(memory_name):
            memory_arn = m['arn']
            print(f"AgentCore Memory already exists: {memory_arn}")
            return memory_arn

    # Create AgentCore Memory.
    response = agentcore_control.create_memory(
        name=memory_name,
        description="Session-scoped conversational memory for NovaMart customer support agents.",
        eventExpiryDuration=7,
        memoryStrategies=[
            {
                'summaryMemoryStrategy': {
                    'name': 'SessionSummarizer',
                    'namespaceTemplates': [
                        '/summaries/{actorId}/{sessionId}/'
                    ]
                }
            }
        ],
        clientToken=str(uuid.uuid4())
    )

    memory = response['memory']
    memory_arn = memory['arn']

    print(f"AgentCore Memory created: {memory_arn}")
    return memory_arn


# ═══════════════════════════════════════════════════════
#  TASK 6 - OBSERVABILITY
# ═══════════════════════════════════════════════════════


def configure_observability(runtime_arn: str) -> None:
    """
    Configure AgentCore Observability:
    - Agent logs -> CloudWatch Logs at INFO level
    - Execution traces -> AWS X-Ray at 100% sampling
    """
    from agent_observability import apply_observability_config

    logging_configuration = {
        "cloudWatchConfig": {
            "logGroupName": config.AGENT_LOG_GROUP,
            "logLevel": "INFO",
            "enabled": True
        },
        "xRayConfig": {
            "enabled": True,
            "samplingRate": 1.0
        }
    }

    try:
        summary = apply_observability_config(
            runtime_arn,
            logging_configuration
        )

        print(f"  CloudWatch log group: {summary['log_group']}")
        print("  CloudWatch logging: INFO (enabled)")
        print("  X-Ray tracing: enabled")
        print("  X-Ray sampling rate: 100%")

    except Exception as e:
        print(f"[Note] Observability configuration skipped: {e}")


# ═══════════════════════════════════════════════════════
#  AGENTCORE GATEWAY DEPLOYMENT  (pre-written - do not modify)
#
#  Production equivalent of in-process @tool functions.
#  Registers Lambda-backed tools on a managed MCP endpoint so tools
#  can be independently deployed, versioned, and discovered at runtime.
#
#  Pattern (from Lesson 11):
#    Local dev  → LambdaGateway + gateway.register_target(...)
#    Production → deploy_agentcore_gateway() using real AWS API
#
#  Requires Lambda tool functions to be deployed separately.
#  Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env
#  to the deployed Lambda function names.
# ═══════════════════════════════════════════════════════

# Lambda function names for gateway tool backends (set in .env after deploying)
_ORDERS_FUNCTION    = os.environ.get('ORDERS_FUNCTION',    f"{config.PROJECT_NAME}-orders-api")
_POLICY_FUNCTION    = os.environ.get('POLICY_FUNCTION',    f"{config.PROJECT_NAME}-policy-api")
_CUSTOMERS_FUNCTION = os.environ.get('CUSTOMERS_FUNCTION', f"{config.PROJECT_NAME}-customers-api")


def _gw_get_function_arn(function_name: str) -> str:
    """Resolve a Lambda function name to its full ARN."""
    lambda_client = boto3.client('lambda', region_name=config.AWS_REGION)
    resp = lambda_client.get_function(FunctionName=function_name)
    return resp['Configuration']['FunctionArn']


def _gw_stack_uuid() -> str:
    """Return the short UUID from the project CloudFormation stack ID.
    Gives the gateway a stable name so re-runs never hit ConflictException."""
    cf = boto3.client('cloudformation', region_name=config.AWS_REGION)
    stacks = cf.describe_stacks(StackName=config.PROJECT_NAME)
    stack_id = stacks['Stacks'][0]['StackId']
    full_uuid = stack_id.split('/')[-1]
    return full_uuid.split('-')[0]


def _gw_wait_for_ready(agentcore_ctrl, gateway_id: str, timeout: int = 120) -> str:
    """Poll until the gateway reaches READY status. Returns the gateway URL."""
    deadline = time.time() + timeout
    first    = True
    while time.time() < deadline:
        gw     = agentcore_ctrl.get_gateway(gatewayIdentifier=gateway_id)
        status = gw['status']
        if status == 'READY':
            if not first:
                print(' ready.')
            return gw.get('gatewayUrl', '')
        if 'FAILED' in status:
            print(f' failed: {status}')
            raise RuntimeError(f"Gateway {gateway_id} entered status {status}")
        if first:
            print('    Gateway provisioning (async — normal AWS behaviour)',
                  end='', flush=True)
            first = False
        print('.', end='', flush=True)
        time.sleep(5)
    raise TimeoutError(f"Gateway {gateway_id} not READY after {timeout}s")


def _gw_get_or_create(agentcore_ctrl, name: str, role_arn: str,
                       instructions: str) -> tuple[str, str]:
    """Create an AgentCore Gateway, or reuse it if it already exists."""
    try:
        gw = agentcore_ctrl.create_gateway(
            name=name,
            roleArn=role_arn,
            protocolType='MCP',
            authorizerType='NONE',
            protocolConfiguration={'mcp': {'instructions': instructions,
                                            'searchType': 'SEMANTIC'}},
        )
        gw_id  = gw['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        print(f'    Status      : {gw["status"]}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    Gateway '{name}' already exists — reusing it.")
        gateways = agentcore_ctrl.list_gateways().get('items', [])
        existing = next((g for g in gateways if g['name'] == name), None)
        if not existing:
            raise RuntimeError(f"Gateway '{name}' not found after ConflictException")
        gw_id  = existing['gatewayId']
        print(f'    Gateway ID  : {gw_id}')
        gw_url = _gw_wait_for_ready(agentcore_ctrl, gw_id)
        print(f'    Gateway URL : {gw_url}')
        return gw_id, gw_url


def _gw_create_target(agentcore_ctrl, gateway_id: str, t: dict,
                       lambda_arn: str) -> None:
    """Register one Lambda target on the gateway. Skips if it already exists."""
    payload = dict(
        gatewayIdentifier=gateway_id,
        name=t['name'],
        description=t['description'],
        targetConfiguration={
            'mcp': {
                'lambda': {
                    'lambdaArn': lambda_arn,
                    'toolSchema': {
                        'inlinePayload': [{
                            'name':        t['tool_name'],
                            'description': t['tool_description'],
                            'inputSchema': {
                                'type': 'object',
                                'properties': {
                                    t['param_name']: {
                                        'type':        'string',
                                        'description': t['param_desc'],
                                    }
                                },
                                'required': [t['param_name']],
                            },
                        }]
                    },
                }
            }
        },
        credentialProviderConfigurations=[
            {'credentialProviderType': 'GATEWAY_IAM_ROLE'}
        ],
    )
    try:
        resp = agentcore_ctrl.create_gateway_target(**payload)
        print(f"    [{resp['status']:12s}] {t['name']} → target {resp['targetId']}")
    except agentcore_ctrl.exceptions.ConflictException:
        print(f"    [already exists] {t['name']} — skipped")


def deploy_agentcore_gateway() -> dict:
    """
    Create an AgentCore Gateway and register the NovaMart tool Lambda targets.

    Production equivalent of the in-process @tool functions defined inside
    build_*_agent(). Each tool becomes a Lambda function registered as a
    gateway target; agents discover tools at runtime via the MCP endpoint —
    no code changes needed when adding or updating tools.

    Uses the same three-step pattern as Lesson 11:
      1. create_gateway  (MCP protocol, SEMANTIC search)
      2. create_gateway_target  (one per Lambda-backed tool)
      3. Agents connect via the returned gateway_url

    Requires Lambda tool functions to be deployed via a separate stack.
    Set ORDERS_FUNCTION, POLICY_FUNCTION, CUSTOMERS_FUNCTION in .env.

    Returns:
        dict with gateway_id, gateway_url, and status.
    """
    agentcore_ctrl = boto3.client('bedrock-agentcore-control',
                                   region_name=config.AWS_REGION)

    try:
        gw_uuid = _gw_stack_uuid()
    except Exception:
        gw_uuid = config.PROJECT_NAME

    gw_name = f"novamart-support-{gw_uuid}"
    print(f"  Calling create_gateway (name: {gw_name})...")
    gateway_id, gateway_url = _gw_get_or_create(
        agentcore_ctrl, gw_name, config.AGENTCORE_ROLE_ARN,
        "NovaMart customer support gateway. Provides order lookup, "
        "policy search, and customer tier tools.",
    )

    targets = [
        {
            'name':             'orders-api',
            'description':      'Look up order details, status, and return eligibility for a customer',
            'function':         _ORDERS_FUNCTION,
            'tool_name':        'check_order_status',
            'tool_description': 'Check order status and return eligibility for a specific order',
            'param_name':       'order_id',
            'param_desc':       'Order ID (e.g. ORD-27176)',
        },
        {
            'name':             'policy-api',
            'description':      'Retrieve return, shipping, and warranty policy text from knowledge bases',
            'function':         _POLICY_FUNCTION,
            'tool_name':        'search_policies',
            'tool_description': 'Search all policy knowledge bases for a customer query',
            'param_name':       'query',
            'param_desc':       'Customer question about returns, shipping, or warranty',
        },
        {
            'name':             'customers-api',
            'description':      'Look up customer tier (Standard or Premium) and account details',
            'function':         _CUSTOMERS_FUNCTION,
            'tool_name':        'get_customer_tier',
            'tool_description': 'Get customer tier and account information by customer ID',
            'param_name':       'customer_id',
            'param_desc':       'Customer ID (e.g. CUST-001)',
        },
    ]

    print(f"\n  Registering {len(targets)} Gateway targets...")
    for t in targets:
        try:
            lambda_arn = _gw_get_function_arn(t['function'])
            _gw_create_target(agentcore_ctrl, gateway_id, t, lambda_arn)
        except Exception as e:
            print(f"    [Skipped] {t['name']}: {e}")

    return {'gateway_id': gateway_id, 'gateway_url': gateway_url, 'status': 'CREATING'}


# ═══════════════════════════════════════════════════════
#  RUNTIME INVOCATION (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def invoke_agent(session_id: str, customer_id: str, user_message: str) -> str:
    """
    Invoke the deployed agent via AgentCore Runtime.
    Pre-written - do not modify.
    """
    enriched_message = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {user_message}"

    response = agentcore_client.invoke_agent_runtime(
        agentRuntimeArn=config.AGENTCORE_RUNTIME_ARN,
        sessionId=session_id,
        inputText=enriched_message,
    )

    full_response = ""
    for event in response.get('completion', []):
        if 'chunk' in event:
            chunk = event['chunk']
            if 'bytes' in chunk:
                full_response += chunk['bytes'].decode('utf-8')

    return full_response


# ═══════════════════════════════════════════════════════
#  DEPLOYMENT ENTRY POINT (pre-written - do not modify)
# ═══════════════════════════════════════════════════════

def deploy_all():
    """Full deployment pipeline. Run after completing all tasks."""
    print("\n" + "="*60)
    print("  Deploying Enterprise Multi-Agent System")
    print("="*60 + "\n")

    print("Step 1/6: Building agent graph...")
    inventory_agent     = build_inventory_agent()
    refund_agent        = build_refund_agent()
    policy_agent        = build_policy_agent()
    communication_agent = build_communication_agent()
    orchestrator = build_orchestrator_agent(
        inventory_agent, refund_agent, policy_agent, communication_agent
    )
    print("  All 5 agents initialized\n")

    print("Step 2/6: Creating Bedrock Guardrail...")
    guardrail_id, guardrail_version = create_guardrail()
    print()

    print("Step 3/6: Deploying to AgentCore Runtime...")
    runtime_arn = deploy_to_agentcore_runtime(orchestrator, guardrail_id, guardrail_version)
    print()

    print("Step 4/6: Configuring Memory...")
    memory_arn = configure_memory(runtime_arn)
    print()

    print("Step 5/6: Configuring Observability...")
    configure_observability(runtime_arn)
    print()

    print("Step 6/6: Deploying AgentCore Gateway...")
    try:
        gw = deploy_agentcore_gateway()
        print(f"  Gateway URL : {gw['gateway_url']}")
        print(f"  Agents connect via MCP at this endpoint — no code changes needed")
    except Exception as e:
        print(f"  [Note] Gateway deployment skipped: {e}")
        print(f"  (Deploy Lambda tool functions and set ORDERS_FUNCTION etc. in .env to enable)")
    print()

    print("="*60)
    print("  Deployment Complete!")
    print("="*60)
    print(f"\n  Add these to your .env file:")
    print(f"  AGENTCORE_RUNTIME_ARN={runtime_arn}")
    print(f"  GUARDRAIL_ID={guardrail_id}")
    print(f"  GUARDRAIL_VERSION={guardrail_version}\n")
    return runtime_arn, guardrail_id


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'deploy':
        deploy_all()

    elif len(sys.argv) > 1 and sys.argv[1] == 'test':
        print("Running local agent test...")
        inventory_agent     = build_inventory_agent()
        refund_agent        = build_refund_agent()
        policy_agent        = build_policy_agent()
        communication_agent = build_communication_agent()
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )

        test_cases = [
            ("CUST-001", "I want to return my wireless headphones from order ORD-27176"),
            ("CUST-002", "What is the return policy for premium customers?"),
            ("CUST-003", "How much would 5 items at $29.99 be with a 10% discount?"),
        ]
        for customer_id, query in test_cases:
            session_id = str(uuid.uuid4())[:8]
            print(f"\n{'─'*60}")
            print(f"Session: {session_id} | Customer: {customer_id}")
            print(f"Query: {query}")
            prompt = f"[Session ID: {session_id}] [Customer ID: {customer_id}] {query}"
            response = orchestrator(prompt)
            print(f"Response: {response}")

    elif len(sys.argv) > 1 and sys.argv[1] == 'chat':
        # ── Interactive terminal chat - educational mode ───────────────────
        W = _C.W

        # ── Welcome banner ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
        print(f"  {_C.ORCH}{_C.BOLD}{'NovaMart -- Multi-Agent Customer Support':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'Strands Agents SDK  +  Amazon Bedrock AgentCore':^{W}}{_C.RESET}")
        print(f"  {_C.GRY}{'=' * W}{_C.RESET}")

        # ── Test customers ────────────────────────────────────────────────
        print()
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.BOLD}Test Customers{_C.RESET}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print(f"  {_C.GRY}{'ID':<10}  {'Name':<18}  {'Tier':<10}  {'Order':<12}  Product{_C.RESET}")
        print(f"  {_C.GRY}{'─'*8}  {'─'*16}  {'─'*8}  {'─'*10}  {'─'*20}{_C.RESET}")
        for cid, name, tier, order, product in [
            ("CUST-001", "Alice Johnson", "Premium",  "ORD-27176", "Sony headphones"),
            ("CUST-002", "Bob Smith",     "Standard", "ORD-28001", "mechanical keyboard"),
            ("CUST-003", "Carol Davis",   "Premium",  "ORD-29001", "laptop"),
            ("CUST-004", "David Lee",     "Standard", "ORD-30001", "phone case"),
        ]:
            tier_col = _C.INV if tier == 'Premium' else _C.GRY
            print(f"  {_C.BOLD}{cid}{_C.RESET}  {name:<18}  "
                  f"{tier_col}{tier:<10}{_C.RESET}  {order}  {product}")
        print(f"  {_C.GRY}{'─' * W}{_C.RESET}")
        print()

        customer_id = (
            input(f"  Enter Customer ID (default: CUST-001): ").strip()
            or "CUST-001"
        )
        session_id  = str(uuid.uuid4())[:8]
        print()
        print(f"  {_C.GRY}Session  : {_C.RESET}{_C.BOLD}{session_id}{_C.RESET}")
        print(f"  {_C.GRY}Customer : {_C.RESET}{_C.BOLD}{customer_id}{_C.RESET}")
        print(f"  {_C.GRY}Type a question and press Enter.  Type 'quit' to exit.{_C.RESET}")
        print()

        # ── Build agents (one line per agent so students see initialisation order)
        print(f"  {_C.GRY}[SYSTEM]  Initializing agent graph...{_C.RESET}")
        inventory_agent     = build_inventory_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  InventoryAgent{_C.RESET}",    flush=True)
        refund_agent        = build_refund_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  RefundAgent{_C.RESET}",       flush=True)
        policy_agent        = build_policy_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  PolicyAgent{_C.RESET}",       flush=True)
        communication_agent = build_communication_agent()
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  CommunicationAgent{_C.RESET}", flush=True)
        orchestrator = build_orchestrator_agent(
            inventory_agent, refund_agent, policy_agent, communication_agent
        )
        print(f"  {_C.GRY}          {_C.OK}[OK]{_C.RESET}{_C.GRY}  Orchestrator{_C.RESET}",      flush=True)
        print(f"  {_C.GRY}[SYSTEM]  All 5 agents ready.{_C.RESET}")
        print()

        # ── Conversation loop ─────────────────────────────────────────────
        while True:
            try:
                user_input = input(
                    f"  {_C.BOLD}You >{_C.RESET} "
                ).strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n  {_C.GRY}Session ended.{_C.RESET}")
                break

            if not user_input:
                continue
            if user_input.lower() in ('quit', 'exit', 'q'):
                print(f"  {_C.GRY}Session ended.{_C.RESET}")
                break

            prompt  = (f"[Session ID: {session_id}] "
                       f"[Customer ID: {customer_id}] {user_input}")
            t0_turn = time.time()

            # ── Install proxy, run orchestrator, restore stdout ────────────
            trace.new_turn()
            sys.stdout = _trace_writer
            try:
                response = orchestrator(prompt)
            finally:
                sys.stdout = _real_stdout   # always restore, even on exception

            elapsed = time.time() - t0_turn

            # ── Resolve the final customer-facing text ────────────────────
            final_state = _read_workflow_state(session_id) or {}
            comm_result = final_state.get('communication_agent', '')
            text = _strip_xml_tags(comm_result or str(response))

            # ── DynamoDB workflow state summary ───────────────────────────
            trace.summary(session_id, elapsed)

            # ── Final customer-facing response ────────────────────────────
            print()
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print(f"  {_C.COM}{_C.BOLD}AGENT RESPONSE{_C.RESET}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            for line in text.splitlines():
                print(f"  {line}")
            print(f"  {_C.GRY}{'=' * W}{_C.RESET}")
            print()

    else:
        print("Usage:")
        print("  python agent_orchestrator.py deploy  # Deploy to AgentCore")
        print("  python agent_orchestrator.py test    # Run automated test cases")
        print("  python agent_orchestrator.py chat    # Interactive terminal chat")
