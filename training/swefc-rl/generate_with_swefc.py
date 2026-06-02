from agent_judge import JudgeAgent
from agent_runner import AgentRunner
from api_sglang import SGLangCompletionClient
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample
from utils import caculte_score, log_runtime_info


async def reward_func(args, sample, **kwargs):
    if not isinstance(sample, Sample):
        raise TypeError("Sample must be an instance of Sample class.")

    if sample.status == Sample.Status.FAILED:
        return {"score": 0}

    reward_score_rule = 1
    reward_score_file = 0
    reward_score_quality = 0

    final_answer_result = sample.final_answer_result

    if final_answer_result["n_citations"] < 1:
        reward_score_rule = 0
    if reward_score_rule == 0:
        score = 0
    else:
        reward_score_file = sample.metadata.get("score_file", 0)
        use_llm_as_judge = True
        if use_llm_as_judge:
            query = sample.prompt[-1]["content"]
            citation_contents = sample.metadata.get("citation_contents", "")
            reward_score_quality = await JudgeAgent().judge(query, citation_contents)
        score = reward_score_rule + reward_score_file + reward_score_quality * 2
    log_runtime_info(
        {
            "reward_score_rule": reward_score_rule,
            "reward_score_file": reward_score_file,
            "reward_score_quality": reward_score_quality,
        }
    )
    return {"score": score}


async def generate(args, sample: Sample, sampling_params, evaluation: bool = False) -> Sample:

    state = GenerateState(args)
    tokenizer = state.tokenizer
    workspace = sample.metadata.get("workspace", "/testbed/")
    prompt_msgs = sample.prompt
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    api_client = SGLangCompletionClient(
        tokenizer=tokenizer,
        sampling_params=sampling_params,
        url=url,
        tool_call_parser=args.sglang_tool_call_parser,
        chat_template_kwargs={"enable_thinking": False},
    )
    agent_runner = AgentRunner(model="swefc-rl-4b", max_iterations=8, api_client=api_client)
    try:
        finish_reason, tokens, prompt_ids, loss_mask, final_response, final_answer_result, runtime_info = (
            await agent_runner.run(prompt_msgs, work_dir=workspace)
        )
    except Exception:
        sample.response = ""
        sample.tokens = [0, 0]  # dummy tokens: 1 prompt + 1 response to satisfy prompt_length >= 1
        sample.loss_mask = [0]
        sample.response_length = 1
        sample.final_answer_result = {"n_citations": 0, "n_broken_lines": 10}
        sample.remove_sample = True
        sample.status = Sample.Status.FAILED
        return sample

    total_length = len(tokens)
    response_length = len(loss_mask)
    prompt_length = len(prompt_ids)

    if total_length > 65536 or prompt_length < 1 or response_length < 1:
        truncated_tokens = tokens[:65536]
        sample.tokens = truncated_tokens
        sample.response = final_response
        sample.loss_mask = [0] * (len(truncated_tokens) - 1)
        sample.response_length = len(truncated_tokens) - 1
        sample.final_answer_result = {"n_citations": 0, "n_broken_lines": 10}
        sample.remove_sample = True
        sample.status = Sample.Status.FAILED
        print(
            f"Sample failed due to length constraints: total_length={total_length}, prompt_length={prompt_length}, response_length={response_length}"
        )
        return sample

    # Clamp response_length to ensure prompt_length >= 1 for downstream padding
    response_length = min(response_length, total_length - 1)
    loss_mask = loss_mask[:response_length]

    sample.tokens = tokens
    sample.response = final_response
    sample.response_length = response_length
    sample.loss_mask = loss_mask

    sample.final_answer_result = final_answer_result
    sample.metadata["n_turn"] = runtime_info.get("n_turn", 0)
    sample.metadata["n_max_parallel_tool_calls"] = runtime_info.get("n_max_parallel_tool_calls", 0)
    sample.metadata["citation_contents"] = runtime_info.get("citation_contents", "")

    if finish_reason == "stop":
        score_details = caculte_score(final_response, sample.label, drop_prefix=workspace)
        sample.metadata["score_file"] = score_details["score_file"]["score"]
        sample.metadata["score_line"] = score_details["score_line"]["score"]
    else:
        sample.metadata["score_file"] = 0
        sample.metadata["score_line"] = 0

    match finish_reason:
        case "length":
            sample.status = Sample.Status.TRUNCATED
        case "abort":
            sample.status = Sample.Status.ABORTED
        case "stop":
            sample.status = Sample.Status.COMPLETED
        case _:
            sample.status = Sample.Status.FAILED

    runtime_info["score_file"] = sample.metadata["score_file"]
    runtime_info["score_line"] = sample.metadata["score_line"]
    if "citation_contents" in runtime_info:
        del runtime_info["citation_contents"]
    log_runtime_info(runtime_info)

    return sample
