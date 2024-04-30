def mutation_prompt_constructor(code, metadata):
    normal_mutation_template = """
    Here is your original code:
    ```
    {}
    ```
    Please try to improve it.
    """

    fix_mutation_template = """
    Here is your original code:
    ```
    {}
    ```
    The code is incorrect.
    And the error is:
    ```
    ```
    Please try to fix it.
    """

    return normal_mutation_template.format(code)


def mutation_respond_extractor(response):
    return response


def mutate(llm_backend, prompt_constructor, codes):
    prompts = [prompt_constructor(code) for code in codes]
    responds = llm_backend.query(prompts)
    return [mutation_respond_extractor(response) for response in responds]
