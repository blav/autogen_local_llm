from model_config import get_config
from llama_cpp import Llama
from autogen.agentchat import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from openai_api import start_openai_api_thread
import chromadb

if __name__ == "__main__":
    config = get_config("OpenHermes-2.5-neural-chat-7B-v3-1-7B")
    
    start_openai_api_thread(Llama(**config))

    config_list= [{
        "base_url": "http://localhost:8000/v1",
        "model": "dontcare",
        "api_key": "dontcare",
    }]

    llm_config = {
        "timeout": 60,
        "cache_seed": None,
        "config_list": config_list,
        "temperature": 0,
    }

    termination_msg = lambda x: isinstance(x, dict) and "TERMINATE" == str(x.get("content", ""))[-9:].upper()

    boss = UserProxyAgent(
        name="Boss",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        system_message="The boss who ask questions and give tasks.",
        code_execution_config=False,  # we don't want to execute code in this case.
        default_auto_reply="Reply `TERMINATE` if the task is done.",
    )

    boss_aid = RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        code_execution_config=False,  # we don't want to execute code in this case.
        retrieve_config={
            "task": "code",
            "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "chunk_token_size": 1000,
            "model": config_list[0]["model"],
            "client": chromadb.PersistentClient(path="/tmp/chromadb"),
            "collection_name": "groupchat",
            "get_or_create": True,
        },
    )

    coder = AssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    pm = AssistantAgent(
        name="Product_Manager",
        is_termination_msg=termination_msg,
        system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    reviewer = AssistantAgent(
        name="Code_Reviewer",
        is_termination_msg=termination_msg,
        system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=llm_config,
    )

    PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."


    def _reset_agents():
        boss.reset()
        boss_aid.reset()
        coder.reset()
        pm.reset()
        reviewer.reset()


    def rag_chat():
        _reset_agents()
        groupchat = GroupChat(
            agents=[boss_aid, coder, pm, reviewer], messages=[], max_round=12, speaker_selection_method="round_robin"
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Start chatting with boss_aid as this is the user proxy agent.
        boss_aid.initiate_chat(
            manager,
            problem=PROBLEM,
            n_results=3,
        )


    def norag_chat():
        _reset_agents()
        groupchat = GroupChat(
            agents=[boss, coder, pm, reviewer],
            messages=[],
            max_round=12,
            speaker_selection_method="auto",
            allow_repeat_speaker=False,
        )
        manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)

        # Start chatting with the boss as this is the user proxy agent.
        boss.initiate_chat(
            manager,
            message=PROBLEM,
        )


    def call_rag_chat():
        _reset_agents()

        # In this case, we will have multiple user proxy agents and we don't initiate the chat
        # with RAG user proxy agent.
        # In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
        # it from other agents.
        def retrieve_content(message, n_results=3):
            boss_aid.n_results = n_results  # Set the number of results to be retrieved.
            # Check if we need to update the context.
            update_context_case1, update_context_case2 = boss_aid._check_update_context(message)
            if (update_context_case1 or update_context_case2) and boss_aid.update_context:
                boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
                _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
            else:
                ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
            return ret_msg if ret_msg else message

        boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

        llm_config = {
            "functions": [
                {
                    "name": "retrieve_content",
                    "description": "retrieve content for code generation and question answering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                            }
                        },
                        "required": ["message"],
                    },
                },
            ],
            "config_list": config_list,
            "timeout": 60,
            "cache_seed": 42,
        }

        for agent in [coder, pm, reviewer]:
            # update llm_config for assistant agents.
            agent.llm_config.update(llm_config)

        for agent in [boss, coder, pm, reviewer]:
            # register functions for all agents.
            agent.register_function(
                function_map={
                    "retrieve_content": retrieve_content,
                }
            )

        groupchat = GroupChat(
            agents=[boss, coder, pm, reviewer],
            messages=[],
            max_round=12,
            speaker_selection_method="random",
            allow_repeat_speaker=False,
        )

        manager_llm_config = llm_config.copy()
        manager_llm_config.pop("functions")
        manager = GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

        # Start chatting with the boss as this is the user proxy agent.
        boss.initiate_chat(
            manager,
            message=PROBLEM,
        )

    rag_chat()

