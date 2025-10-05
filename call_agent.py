from agents.agent import initialize_agent
from agents.utils import ModelType

agent = initialize_agent(model=ModelType.GPT, role="sales_manager", use_platform_store=True)