from extraction.run_extraction import run_extraction
from evaluation.run_evaluation import run_evaluation
from extraction.run_extraction_for_analysis import run_extraction_for_analysis
from extraction.run_extraction_for_prompt import run_extraction_for_prompt
from evaluation.run_evaluation_for_prompt import run_evaluation_for_prompt
import yaml


def run_all_extraction():
    with open(f"config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        model_dict = config['models']
    for dataset in ["aws", "azure", "gcp"]:
        for model_abbr, model_name in model_dict.items():
            for prompt_type in ["0", "1"]:
                run_extraction(
                    dataset=dataset,
                    model_abbr=model_abbr,
                    model_name=model_name,
                    prompt_type=prompt_type,
                    round_mark="ext",
                )

def run_all_extraction_for_prompt():
    # with open(f"config.yaml", 'r') as f:
    #     config = yaml.safe_load(f)
    #     model_dict = config['models']
    model_dict = {'gpt-3.5': 'gpt-3.5-turbo'}
    for dataset in ["aws"]:
        for model_abbr, model_name in model_dict.items():
            for prompt_strategy_type in ["basic-fs", "basic-zs", "categ-zs"]:
                run_extraction_for_prompt(
                    dataset=dataset,
                    model_abbr=model_abbr,
                    model_name=model_name,
                    prompt_strategy_type=prompt_strategy_type,
                )

def run_all_extraction_for_analysis():
    run_extraction_for_analysis(
        dataset="aws",
        model_abbr="gemini-2.5",
        model_name="gemini-2.5-pro",
        prompt_type="1",
        round_mark="anl",
    )
    run_extraction_for_analysis(
        dataset="azure",
        model_abbr="gemini-2.0",
        model_name="gemini-2.0-flash",
        prompt_type="1",
        round_mark="anl",
    )


def run_all_evaluation():
    with open(f"config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        model_dict = config['models']
    # model_dict = {'gpt-3.5': 'gpt-3.5-turbo'}
    for model_abbr, model_name in model_dict.items():
        for dataset in ["aws", "azure", "gcp"]:
            for prompt_type in ["0", "1"]:
                run_evaluation( 
                    round_mark="eval",  
                    dataset=dataset,
                    model_abbr=model_abbr,
                    model_name=model_name,
                    prompt_type=prompt_type,
                    eval_fields=[
                        'service_name', 'start_time', 'end_time', 'timezone',
                        'service_category', 'user_symptom_category',
                        'user_symptom', 'root_cause', 'root_cause_category'
                    ],
                    eval_methods=["EM", "TK", "BS"],  
                )

def run_all_evaluation_for_prompt():
    # with open(f"config.yaml", 'r') as f:
    #     config = yaml.safe_load(f)
    #     model_dict = config['models']
    model_dict = {'gpt-3.5': 'gpt-3.5-turbo'}
    for model_abbr, model_name in model_dict.items():
        for dataset in ["aws"]:
            for prompt_strategy_type in ["basic-fs", "basic-zs", "categ-zs"]:
                run_evaluation_for_prompt( 
                    round_mark="pmpt",  
                    dataset=dataset,
                    model_abbr=model_abbr,
                    model_name=model_name,
                    prompt_strategy_type=prompt_strategy_type,
                    eval_fields=[
                        'service_name', 'location', 'start_time', 'end_time', 'timezone',
                        'service_category', 'user_symptom_category', 'user_symptom'
                    ],
                    eval_methods=["EM", "TK", "BS"],  
                )


if __name__ == "__main__":
    # run_all_extraction()
    # run_all_evaluation()

    # run_all_extraction_for_analysis()

    # run_all_extraction_for_prompt()
    run_all_evaluation_for_prompt()