import os
from pathlib import Path

from compare_location_runs import compare_runs
from split_user_location import run


def require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def optional_env(name: str, default: str) -> str:
    return os.getenv(name, default).strip() or default


def load_dotenv_if_available(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        import importlib

        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")
        load_dotenv(dotenv_path=str(env_path))
    except (ImportError, AttributeError):
        pass


def main() -> None:
    scripts_dir = Path(__file__).resolve().parent
    project_root = scripts_dir.parent

    load_dotenv_if_available(project_root)

    input_csv = optional_env("INPUT_CSV", "processed_tourism_reviews.csv")
    output_a = optional_env("OUTPUT_A_CSV", "processed_tourism_reviews_country_other_openrouter.csv")
    output_b = optional_env("OUTPUT_B_CSV", "processed_tourism_reviews_country_other_groq.csv")
    mismatches_out = optional_env("MISMATCHES_OUT", "location_llm_mismatches.csv")

    input_path = str((project_root / input_csv).resolve())
    output_a_path = str((project_root / output_a).resolve())
    output_b_path = str((project_root / output_b).resolve())
    mismatches_path = str((project_root / mismatches_out).resolve())

    provider_a_name = optional_env("PROVIDER_A_NAME", "openrouter")
    provider_b_name = optional_env("PROVIDER_B_NAME", "groq")

    provider_a_key = require_env("OPENROUTER_API_KEY")
    provider_a_base = optional_env("OPENROUTER_API_BASE_URL", "https://openrouter.ai/api/v1")
    provider_a_model = optional_env("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")

    provider_b_key = require_env("GROQ_API_KEY")
    provider_b_base = optional_env("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1")
    provider_b_model = optional_env("GROQ_MODEL", "llama-3.3-70b-versatile")

    print("Running provider A...")
    print(f"Provider A: {provider_a_name}, model={provider_a_model}")
    run(
        input_csv=input_path,
        output_csv=output_a_path,
        location_column="User_Location",
        use_llm=True,
        llm_api_key=provider_a_key,
        llm_api_base_url=provider_a_base,
        llm_model=provider_a_model,
        llm_timeout=20,
    )

    print("Running provider B...")
    print(f"Provider B: {provider_b_name}, model={provider_b_model}")
    run(
        input_csv=input_path,
        output_csv=output_b_path,
        location_column="User_Location",
        use_llm=True,
        llm_api_key=provider_b_key,
        llm_api_base_url=provider_b_base,
        llm_model=provider_b_model,
        llm_timeout=20,
    )

    print("Comparing outputs...")
    compare_runs(
        file_a=output_a_path,
        file_b=output_b_path,
        label_a=provider_a_name,
        label_b=provider_b_name,
        mismatches_out=mismatches_path,
    )


if __name__ == "__main__":
    main()
