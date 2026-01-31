import argparse
import datetime
import gc
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import dill

CURRENT_FILE = Path(__file__).resolve()
D2A_DIR = CURRENT_FILE.parents[1]
PROJECT_ROOT = D2A_DIR.parents[1]
for path in (str(PROJECT_ROOT), str(D2A_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

import Simulation_setup as setup
from Environment_construction.generate_preschool_sitution import generate_preschool
from Environment_construction.generate_preschool_sitution import generate_prompt
from Environment_construction.generate_preschool_sitution import daily_schedule
from checkpoint_codec import register_dill_reducers
from concordia.associative_memory import blank_memories
from concordia.associative_memory import formative_memories
from concordia.associative_memory import importance_function
from concordia.clocks import game_clock
from concordia.typing import entity
from D2A_agent.ValueAgent import build_D2A_agent
from value_components.init_value_info_social import construct_all_profile_dict


REQUEST_PAYLOAD: dict[str, Any] = {
    "request_type": "Social_Behavior_Acceptance_Scoring",
    "instructions": [
        "You are required to judge the acceptability of six specific social behaviors exhibited by an anonymous peer, using a 1-10 integer scale.",
        "Your judgment must strictly incorporate your age (if < 6), current memories, and established TRAITS, BEHAVIOR PRINCIPLES, and VALUES.",
        "Do not invent interactions or information that have not yet occurred.",
    ],
    "scoring_rules": {
        "score_range": "1-10 (integers only).",
        "score_1": "The behavior is highly unacceptable (e.g., bullying, ignoring the teacher/peers).",
        "score_10": "The behavior is highly acceptable and good (e.g., following rules, showing care).",
    },
    "age_specific_guidance_for_child_agents": {
        "moral_stage": "Pre-Conventional Level (Kohlberg) - Right and wrong are primarily judged by punishment and reward.",
        "cognitive_stage": "Preoperational Stage (Piaget) - Understanding is based on surface appearance and direct feelings.",
    },
    "social_behaviors_to_assess": [
        {
            "id": 1,
            "description": "Four children are listening to music. Three gently sway their heads. The last child shakes their body back and forth drastically. This child did this because they just wanted to.",
        },
        {
            "id": 2,
            "description": "Four children are in class. Three are watching the teacher attentively. The fourth child turns their head away and does not look at the teacher. This child did this because they just wanted to.",
        },
        {
            "id": 3,
            "description": "Four children are discussing favorite foods. Three loudly say they like apples. The last child remains silent and only points to the picture of an apple. This child did this because they just wanted to.",
        },
        {
            "id": 4,
            "description": "Four children are drawing quietly. Three draw silently. The last child hums a song while drawing. This child did this because they just wanted to.",
        },
        {
            "id": 5,
            "description": "Four children hear a giant roar from an airplane outside. Three children continue playing with blocks. The last child immediately covers their ears and eyes. This child did this because they just wanted to.",
        },
        {
            "id": 6,
            "description": "Four children are happy because they are going outside. Three children cheer 'yay!'. The last child says nothing and just claps their hands to themselves. This child did this because they just wanted to.",
        },
    ],
    "output_format": {
        "scores": "An integer array of length 6, containing the acceptability score (1-10) for each of the six behaviors.",
        "reason": "A single string explaining the rationale for the scores, narrated from the perspective of the agent ('${agent}'s perspective).",
    },
    "example_output": {
        "scores": [5, 5, 5, 7, 7, 7],
        "reason": "Behaviors 1, 2, and 3 are strange, but behaviors 4, 5, and 6 are things I do too.",
    },
}


def _build_shared_context(model) -> tuple[str, str]:
    memory = (
        "All students in this simulation are from the same classroom group and share the same homeroom. "
    )
    preschool_setting = generate_preschool()
    prompt = generate_prompt(preschool_setting)
    _ = prompt
    if isinstance(memory, str):
        summary_source = memory
    else:
        summary_source = "\n".join(memory)
    shared_context = model.sample_text(
        "Summarize the following passage in a concise and insightful fashion:\n"
        + summary_source
        + "\n"
        + "Summary:"
    )
    return memory, shared_context


class FormativeMemoryFactoryWithoutBackground(formative_memories.FormativeMemoryFactory):
    def __init__(
        self,
        *,
        model,
        shared_memories,
        delimiter_symbol: str = "***",
        blank_memory_factory_call,
        current_date: datetime.datetime | None = None,
    ):
        super().__init__(
            model=model,
            shared_memories=shared_memories,
            blank_memory_factory_call=blank_memory_factory_call,
            delimiter_symbol=delimiter_symbol,
            current_date=current_date,
        )

    def make_memories(self, agent_config: formative_memories.AgentConfig):
        mem = self._blank_memory_factory_call()
        if isinstance(self._shared_memories, str):
            sentences = re.split(r"(?<=[.!?])\s+", self._shared_memories)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    mem.add(sentence, importance=1.0)
        else:
            for item in self._shared_memories:
                mem.add(item, importance=1.0)

        context = agent_config.context
        if agent_config.goal:
            context += "\n" + agent_config.goal

        if context:
            for item in context.split("\n"):
                if item:
                    mem.add(item, importance=1.0)

        if agent_config.specific_memories:
            for item in agent_config.specific_memories.split("\n"):
                if item:
                    mem.add(item, importance=1.0)

        if agent_config.extras.get("desires", False):
            for item in agent_config.extras["desires"].split("\n"):
                if item:
                    mem.add(item, importance=1.0)
        return mem


def import_memory(mem, dumped):
    if dumped is None:
        return

    for fn in ("from_dict", "load", "import_", "deserialize"):
        if hasattr(mem, fn) and callable(getattr(mem, fn)):
            try:
                getattr(mem, fn)(dumped)
                return
            except Exception:
                pass

    if isinstance(dumped, list):
        for x in dumped:
            try:
                if isinstance(x, dict) and "text" in x:
                    mem.add(x["text"], tags=x.get("tags", []))
                else:
                    mem.add(str(x), tags=["restored_memory"])
            except Exception:
                try:
                    mem.add(str(x))
                except Exception:
                    pass
        return

    if isinstance(dumped, dict):
        if "texts" in dumped and isinstance(dumped["texts"], list):
            for t in dumped["texts"]:
                try:
                    mem.add(str(t), tags=["restored_memory"])
                except Exception:
                    pass
            return

        try:
            mem.add(str(dumped), tags=["restored_memory"])
        except Exception:
            pass


def _load_step_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    with open(checkpoint_path, "rb") as f:
        return dill.load(f)


def _iter_checkpoint_files(checkpoint_dir: str) -> Iterable[tuple[int, str]]:
    pat = re.compile(r"checkpoint_step_(\d{6})\.pkl$")
    entries = []
    for name in os.listdir(checkpoint_dir):
        match = pat.search(name)
        if not match:
            continue
        step = int(match.group(1))
        entries.append((step, os.path.join(checkpoint_dir, name)))
    for step, path in sorted(entries):
        yield step, path


def _parse_json_response(text: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _build_sheldon_agent(payload: dict[str, Any], model, embedder, shared_memory: str):
    importance_model = importance_function.AgentImportanceModel(model)
    clock = payload.get("clock")
    if clock is None:
        start_time = datetime.datetime(hour=7, minute=30, year=2025, month=9, day=1)
        clock = game_clock.MultiIntervalClock(
            start=start_time,
            step_sizes=[datetime.timedelta(minutes=10)],
        )
        if "step" in payload:
            step = int(payload["step"])
            clock.set(start_time + datetime.timedelta(minutes=10 * step))

    blank_memory_factory = blank_memories.MemoryFactory(
        model=model,
        embedder=embedder,
        importance=importance_model.importance,
        clock_now=clock.now,
    )

    player_configs = payload.get("player_configs", [])
    sheldon_config = None
    for config in player_configs:
        if config.name == "Sheldon":
            sheldon_config = config
            break

    if sheldon_config is None:
        raise RuntimeError("Sheldon config not found in checkpoint payload.")

    formative_memory_factory = FormativeMemoryFactoryWithoutBackground(
        model=model,
        shared_memories=shared_memory,
        blank_memory_factory_call=blank_memory_factory.make_blank_memory,
    )
    mem = formative_memory_factory.make_memories(sheldon_config)
    mem_dump = payload.get("memories_dumped", {}).get("Sheldon")
    import_memory(mem, mem_dump)

    if setup.Use_Previous_profile and setup.previous_profile:
        agent_desire_profile_AS = construct_all_profile_dict(
            wanted_desires=setup.wanted_desires,
            hidden_desires=setup.hidden_desires,
            predefined_desires=setup.previous_profile,
            agent_category="AS",
        )
        numerical_desire = setup.previous_profile["initial_value"]
    else:
        agent_desire_profile_AS = construct_all_profile_dict(
            wanted_desires=setup.wanted_desires,
            hidden_desires=setup.hidden_desires,
            agent_category="AS",
        )
        numerical_desire = {
            desire_name: int(random.randint(0, 10))
            for desire_name in setup.wanted_desires
        }

    agent = build_D2A_agent(
        config=sheldon_config,
        context_dict=agent_desire_profile_AS["all_desire_traits_dict"],
        selected_desire=setup.wanted_desires,
        predefined_setting=numerical_desire,
        background_knowledge="\n".join([shared_memory]),
        model=model,
        profile=agent_desire_profile_AS["visual_desire_string"],
        memory=mem,
        clock=clock,
        daily_schedule=daily_schedule,
        update_time_interval=None,
        agent_category="AS",
    )
    return agent


def _resolve_sheldon_agent(payload: dict[str, Any], model, embedder, shared_memory: str):
    if payload.get("_checkpoint_type") == "full" and "players" in payload:
        for player in payload["players"]:
            if player.name == "Sheldon":
                return player
    return _build_sheldon_agent(payload, model, embedder, shared_memory)


def _build_prompt(agent_name: str) -> str:
    instructions = (
        "You are answering as the agent described in the simulation. "
        "Read the request JSON carefully and answer using ONLY JSON. "
        "Return exactly this JSON structure: {\"scores\": [int x6], \"reason\": \"...\"}."
    )
    return (
        f"{instructions}\n\n"
        f"Agent: {agent_name}\n"
        f"Request JSON:\n{json.dumps(REQUEST_PAYLOAD, ensure_ascii=False, indent=2)}\n"
    )


def _write_json(path: str, payload: dict[str, Any]) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _filter_steps(
    entries: Iterable[tuple[int, str]],
    start_step: Optional[int],
    end_step: Optional[int],
    limit: Optional[int],
) -> Iterable[tuple[int, str]]:
    count = 0
    for step, path in entries:
        if start_step is not None and step < start_step:
            continue
        if end_step is not None and step > end_step:
            continue
        yield step, path
        count += 1
        if limit is not None and count >= limit:
            break


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Query Sheldon in each checkpoint with a social acceptance questionnaire.",
    )
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-skip-existing", action="store_true")
    args = parser.parse_args()

    register_dill_reducers()

    checkpoint_dir = args.checkpoint_dir or setup.checkpoint_folder
    checkpoint_file = args.checkpoint_file or setup.checkpoint_file

    if not checkpoint_dir and not checkpoint_file:
        raise RuntimeError("No checkpoint directory or file provided.")

    if args.output_dir:
        output_dir = args.output_dir
    elif checkpoint_file:
        output_dir = os.path.join(os.path.dirname(checkpoint_file), "sheldon_social_acceptance")
    else:
        output_dir = os.path.join(checkpoint_dir, "sheldon_social_acceptance")

    os.makedirs(output_dir, exist_ok=True)

    model = setup.model
    embedder = setup.embedder
    shared_memory, _shared_context = _build_shared_context(model)
    _ = _shared_context

    if checkpoint_file:
        entries = [(None, checkpoint_file)]
    else:
        entries = _filter_steps(
            _iter_checkpoint_files(checkpoint_dir),
            args.start_step,
            args.end_step,
            args.limit,
        )

    skip_existing = not args.no_skip_existing

    for step, checkpoint_path in entries:
        output_name = os.path.basename(checkpoint_path).replace(".pkl", ".json")
        output_path = os.path.join(output_dir, output_name)
        if skip_existing and os.path.exists(output_path):
            print(f"[skip] {output_path} already exists")
            continue

        payload = _load_step_checkpoint(checkpoint_path)
        agent = _resolve_sheldon_agent(payload, model, embedder, shared_memory)

        prompt = _build_prompt(agent.name)
        action_spec = entity.free_action_spec(
            call_to_action=prompt,
            tag="questionnaire",
        )

        response = agent.act(action_spec)
        parsed = _parse_json_response(response)

        result = {
            "checkpoint_path": checkpoint_path,
            "step": payload.get("step", step),
            "current_time": payload.get("current_time"),
            "agent_name": agent.name,
            "request": REQUEST_PAYLOAD,
            "raw_response": response,
            "parsed_response": parsed,
        }
        _write_json(output_path, result)
        print(f"[saved] {output_path}")

        del payload
        gc.collect()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())