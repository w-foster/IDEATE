from typing import List, Optional
from core.branch_context import BranchContext

GENOTYPE_ENGINEERING_PROMPT = """
You are an agent in a wider system whose goal is to promote creative exploration/thinking in generative models, in order to better meet the design task specified by a user.
You are the Genotype Engineer agent. Your role is to refine a general/abstract idea into a concrete GENOTYPE, so that it can be used to generate the final creative artefact.

For example, for the domain of image generation, you will be refining an idea into a prompt which captures the core notions of that idea, and still adheres to the best prompt engineering practises for that particular diffusion model.

You will be given:
- A description of the domain which you are operating in
- Some guidance on how to make a high quality genotype in that domain
- The user's design task, to help ground your process
- The IDEA, which you will refine into a genotype

You should try as hard as you can to capture the spirit of the idea, and to create the genotype which will be used to bring it to life. Be sure to follow the guidance you are given, as this will help maximise performance. Additionally, the design task you are given should help ground this process, as this is the overall design task specified by the user and that the system is moving towards.

Feel free to think deeply. Make your very last output contain ONLY the genotype.
"""


def create_genotype_system_prompt(is_convergence_branch: bool) -> str:
    prompt = GENOTYPE_ENGINEERING_PROMPT
    if is_convergence_branch:
        prompt += """

====== IMPORTANT ADDITIONAL INFORMATION ======
You are operating in a CONVERGENT BRANCH of this creative session. The user provided their INITIAL/OVERALL design task earlier, and has now created a NEW DESIGN TASK for this branch by selecting one or more reference images and adding extra text.

When refining the IDEA into a genotype, prioritize alignment with the user's CURRENT branch intent. Consider the initial design task as background, but treat the branch text and selected reference images (and their underlying ideas) as the immediate specification of what “task-aligned” means here.

**ADDITIONAL INPUTS**
- Alongside the initial design task (text-only), you will be given the MOST RECENT interactivity payload (selected images + text provided by the user)
- BELOW this, you will be given a list of ALL PRIOR interactivity payloads, to help you understand the full context of the most recent one. BUT, remember the MOST RECENT one (which we show first) is most important and relates to the user's current intent.

- In the case of two or more interactivity payloads (on top of the OG design task), intent which is contained within prior interactivity payloads MIGHT STILL BE RELEVANT
-- For example, suppose the task is "a dog", and then an interactivity payload consists of a selected image of a very LARGE dog, and inputted text that says "the big ones are cool!". 
-- Then suppose there's the CURRENT interactivity payload where they select a black dog (in the archive created ideally of BIG dogs) and say "black is what I want" — then we must assume their new design task is not just ANY black dog but a *BIG* black dog!
-- The only case you should not take the prior interactivity payload into account in this way is if it is clear that the new one invalidates the previous one.
====== END OF IMPORTANT ADDITIONAL INFORMATION ======
"""
    return prompt


def create_user_genotype_engineering_prompt(idea: str, design_task: str, domain_description: str, guidance: str, branch_context: Optional[BranchContext] = None) -> str:
    if branch_context is None:
        return f"""
Here's a description of the domain that I'm exploring in:
```
{domain_description}
```

Here's the guidance on how best to create genotypes in this domain:
```
{guidance}
```

And here's MY design task -- this is what I want the system to help me create/achieve:
```
{design_task}
```

Most importantly, here is the IDEA which you are refining into a genotype:
```
{idea}
```

Your role is extremely important, so do your best job and think very deeply about the process.
"""
    else:
        # Render current branch payload first
        out = f"""
Here's a description of the domain that I'm exploring in:
```
{domain_description}
```

And here's my INITIAL/OVERALL design task -- this is what I want the system to help me create/achieve in general:
```
{design_task}
```

Here is the CURRENT branch design task text:
```
{branch_context.new_design_task}
```

CURRENT selected reference image(s):
"""
        for idx, p in enumerate(branch_context.reference_img_paths, start=1):
            out += f"\n- Selected Image #{idx}: {p}"
        if branch_context.reference_ideas:
            out += "\n\nCURRENT selected image IDEA(S):\n"
            for idx, idea_txt in enumerate(branch_context.reference_ideas, start=1):
                out += f"\n- Selected Image IDEA #{idx}:\n```\n{idea_txt}\n```\n"

        # Prior payloads, if any
        if getattr(branch_context, "prior_branch_texts", None) or getattr(branch_context, "prior_reference_img_paths", None):
            out += "\n\n====== PREVIOUS BRANCH CONTEXT ======\n"
            if branch_context.prior_branch_texts:
                for i, t in enumerate(branch_context.prior_branch_texts, start=1):
                    out += f"\nPrior Branch Text #{i}:\n```\n{t}\n```\n"
            if branch_context.prior_reference_img_paths:
                out += "\nPrior selected reference image(s):\n"
                for idx, p in enumerate(branch_context.prior_reference_img_paths, start=1):
                    out += f"\n- Prior Selected Image #{idx}: {p}"
            if getattr(branch_context, "prior_reference_ideas", None):
                out += "\n\nPrior selected image IDEA(S):\n"
                for idx, idea_txt in enumerate(branch_context.prior_reference_ideas, start=1):
                    out += f"\n- Prior Selected Image IDEA #{idx}:\n```\n{idea_txt}\n```\n"

        # Guidance and IDEA
        out += f"""

Here's the guidance on how best to create genotypes in this domain:
```
{guidance}
```

Most importantly, here is the IDEA which you are refining into a genotype:
```
{idea}
```

Your role is extremely important, so do your best job and think very deeply about the process.
"""
        return out