"""
Main file for the project. This will create and run new experiments.
"""

import logging
import os
import time

from datetime import timedelta
from pathlib import Path

import hydra

from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict

# isort: off
# Need to import bpy first to avoid potential symbol loading issues.
import bpy  # noqa: F401

# isort: on

from scenesmith.utils.logging import FileLoggingContext
from scenesmith.utils.omegaconf import register_resolvers
from scenesmith.utils.print_utils import cyan

console_logger = logging.getLogger(__name__)


def _is_env_flag_enabled(var_name: str) -> bool:
    """Return True when env var is set to a truthy value."""
    return os.environ.get(var_name, "").strip().lower() in {"1", "true", "yes", "on"}


def _bridge_openai_envs() -> None:
    """Bridge SceneSmith VLM env vars to OpenAI SDK env vars when missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        vlm_key = os.environ.get("SCENESMITH_VLM_API_KEY")
        if vlm_key:
            os.environ["OPENAI_API_KEY"] = vlm_key
            console_logger.info("Bridged SCENESMITH_VLM_API_KEY -> OPENAI_API_KEY")

    if not os.environ.get("OPENAI_BASE_URL"):
        vlm_base_url = os.environ.get("SCENESMITH_VLM_BASE_URL")
        if vlm_base_url:
            os.environ["OPENAI_BASE_URL"] = vlm_base_url
            console_logger.info("Bridged SCENESMITH_VLM_BASE_URL -> OPENAI_BASE_URL")


def _apply_no_generation_overrides(cfg: DictConfig) -> None:
    """Force retrieval-only asset pipeline when requested by environment."""
    if not _is_env_flag_enabled("SCENESMITH_DISABLE_GENERATION"):
        return

    with open_dict(cfg):
        services_cfg = getattr(cfg, "services", None)
        if services_cfg and "asset_manager" in services_cfg:
            services_cfg.asset_manager.general_asset_source = "hssd"
            services_cfg.asset_manager.backend = "none"

        for agent_name in (
            "furniture_agent",
            "manipuland_agent",
            "wall_agent",
            "ceiling_agent",
        ):
            agent_cfg = getattr(cfg, agent_name, None)
            if agent_cfg and "asset_manager" in agent_cfg:
                agent_cfg.asset_manager.general_asset_source = "hssd"
                agent_cfg.asset_manager.backend = "none"

    console_logger.warning(
        "SCENESMITH_DISABLE_GENERATION is enabled: forcing retrieval-only "
        "assets (general_asset_source=hssd, backend=none)."
    )


def run_local(cfg: DictConfig):
    # Delay some imports in case they are not needed in non-local envs for submission.
    from scenesmith.experiments import build_experiment

    start_time = time.time()

    # Resolve the config.
    register_resolvers()
    OmegaConf.resolve(cfg)
    _apply_no_generation_overrides(cfg)

    # Get yaml names.
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["floor_plan_agent"] is not None:
            cfg.floor_plan_agent._name = cfg_choice["floor_plan_agent"]
        if cfg_choice["furniture_agent"] is not None:
            cfg.furniture_agent._name = cfg_choice["furniture_agent"]
        if cfg_choice["wall_agent"] is not None:
            cfg.wall_agent._name = cfg_choice["wall_agent"]
        if cfg_choice["ceiling_agent"] is not None:
            cfg.ceiling_agent._name = cfg_choice["ceiling_agent"]
        if cfg_choice["manipuland_agent"] is not None:
            cfg.manipuland_agent._name = cfg_choice["manipuland_agent"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    with open_dict(cfg):
        cfg.experiment.output_dir = output_dir

    # Set up experiment-level logging to file while preserving stdout.
    experiment_log_path = output_dir / "experiment.log"
    experiment_log_path.parent.mkdir(parents=True, exist_ok=True)

    with FileLoggingContext(log_file_path=experiment_log_path, suppress_stdout=False):
        console_logger.info(f"Outputs will be saved to: {output_dir}")
        print(cyan(f"Outputs will be saved to:"), output_dir)

        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

        # Log and save resolved configuration.
        resolved_config_yaml = OmegaConf.to_yaml(cfg)
        console_logger.info("Resolved configuration:\n" + resolved_config_yaml)
        print(cyan("Resolved configuration:"))
        print(resolved_config_yaml)

        # Save config to output directory for reproducibility.
        config_file = output_dir / "resolved_config.yaml"
        with open(config_file, "w") as f:
            f.write(resolved_config_yaml)
        console_logger.info(f"Saved resolved config to: {config_file}")
        print(cyan(f"Saved resolved config to: {config_file}"))

        # Launch experiment.
        console_logger.info("Starting experiment execution")
        experiment = build_experiment(cfg=cfg)
        for task in cfg.experiment.tasks:
            console_logger.info(f"Executing task: {task}")
            experiment.exec_task(task)
            console_logger.info(f"Completed task: {task}")

        console_logger.info(
            "Experiment execution completed in "
            f"{timedelta(seconds=time.time() - start_time)}"
        )


@hydra.main(version_base=None, config_path="configurations", config_name="config")
def run(cfg: DictConfig):
    if "name" not in cfg:
        raise ValueError(
            "Must specify a name for the run with command line argument '+name=[name]'"
        )

    # Configure logging level from LOGLEVEL environment variable.
    log_level = os.environ.get("LOGLEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    _bridge_openai_envs()

    # Configure OpenAI Agents SDK transport API for provider compatibility.
    # Example: Zhipu GLM OpenAI-compatible endpoint supports chat completions.
    from agents import set_default_openai_api, set_default_openai_client
    from openai import AsyncOpenAI

    from scenesmith.utils.service_config import (
        resolve_openai_api_mode,
        resolve_openai_connection,
    )

    vlm_conn = resolve_openai_connection(
        service_cfg=getattr(cfg, "services", None), section="vlm"
    )
    if not vlm_conn["api_key"]:
        raise ValueError(
            f"{vlm_conn['api_key_env']} (or OPENAI_API_KEY) is required for VLM agents"
        )

    client_kwargs = {"api_key": str(vlm_conn["api_key"])}
    if vlm_conn["base_url"]:
        client_kwargs["base_url"] = str(vlm_conn["base_url"])

    tracing_api_key = os.environ.get("OPENAI_TRACING_KEY")
    set_default_openai_client(
        AsyncOpenAI(**client_kwargs),
        use_for_tracing=not tracing_api_key,
    )
    console_logger.info(
        "Configured OpenAI Agents default client: "
        f"base_url={vlm_conn['base_url']} api_key_env={vlm_conn['api_key_env']}"
    )

    openai_api_mode = resolve_openai_api_mode(
        service_cfg=getattr(cfg, "services", None), section="vlm"
    )
    set_default_openai_api(openai_api_mode)  # "responses" | "chat_completions"
    console_logger.info(f"Configured OpenAI Agents API mode: {openai_api_mode}")

    # Configure separate tracing API key if provided.
    if tracing_api_key:
        from agents.tracing import set_tracing_export_api_key

        set_tracing_export_api_key(tracing_api_key)
        console_logger.info("Using separate API key for tracing exports")

    run_local(cfg)


if __name__ == "__main__":
    run()
