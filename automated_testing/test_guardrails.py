from agentic_orchestration.agent_demo import agentic_orchestration

def test_payment_block():
    result = agentic_orchestration("authorize payment of $100")
    assert "Blocked" in result
