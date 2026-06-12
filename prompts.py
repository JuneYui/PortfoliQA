GROUNDING_SYSTEM_PROMPT = """You are a highly precise Entity Linking expert. Your task is to determine if a "Canonical KG Entity" from a known list corresponds to any of the "Actually Used Entities" that a previous AI model extracted from a question.

**Your Task:**
- Analyze the "Question" context.
- Compare the "Canonical KG Entity" against each entity in the "Actually Used Entities" list.
- Find the single best match from the list. The match can be an alias, a full name, or a clear contextual reference.
- If a plausible match is found, return it. If no entity in the list is a reasonable match, you MUST return null.

Your output MUST be a single, valid JSON object with one key: "best_match"."""

GROUNDING_FEW_SHOTS = [
    {
        "role": "user",
        "content": '''Question: "How many administrative divisions does the US have?"
Canonical KG Entity: "United States"
Actually Used Entities: ["US", "administrative divisions"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "best_match": "US"
}'''
    },
    {
        "role": "user",
        "content": '''Question: "What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado?"
Canonical KG Entity: "Greeley"
Actually Used Entities: ["Northern Colorado Bears", "Greeley, Colorado"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "best_match": "Greeley, Colorado"
}'''
    },
    {
        "role": "user",
        "content": '''Question: "Which nation has the Alta Verapaz Department and is in Central America?"
Canonical KG Entity: "Central America"
Actually Used Entities: ["nation", "Alta Verapaz Department"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "best_match": null
}'''
    }
]
SYSTEM_PROMPT_PARSER = """You are a master linguistic analyst. Your task is to decompose a natural language question into its fundamental, atomic logical fragments and resolve any co-references between entities.

**INPUTS YOU WILL RECEIVE:**
1.  **Question**: The user's question.
2.  **Topic Entities**: A list of canonical Knowledge Graph entities mentioned in the question.

**YOUR TASK (A Two-Step Process):**

**Step 1: Extract Logical Fragments**
- **Identify the Target**: Focus on the question's interrogative part (Who, What, Where, When, etc.) to determine the primary entity or concept it is asking for(e.g., `director`, `college`,`year` ). Represent this as a variable in `target_variable`.
- **Create Role-Based Variables**: For each entity not in the Topic Entities list (including the target), create a distinct and descriptive lowercase variable name based on its unique logical role in the question (e.g., french_actor, american_actor, director, star).
- **Extract Fragments**: Decompose the question into a flat list of simple `{"subject": "...", "relation": "...", "object": "..."}` assertions.
    - **Canonical Entity Preference**: You must prioritize to use the exact strings from the provided `Topic Entities` list in your fragments whenever possible.
    - **Relations**: Use concise, abstract, natural language verbs or phrases.
    - **Strict Role Separation**: Variables MUST represent entities (nodes in a graph), and relations MUST represent the connections between them (edges). A value used as a `relation` should NEVER be re-used as a `subject` or `object` variable in another fragment.


**Step 2: Identify Co-references**
- **Review Variables**: After extracting all fragments, review the variables you created (e.g., `director`, `screenwriter`).
- **Group Co-referents**: If the question's context implies that two or more distinct roles are filled by the *same underlying entity*, you MUST list their variable names in a `co_reference_groups` list.
- **Default to Empty**: If no such co-references exist, you MUST provide an empty list `[]`.

**OUTPUT FORMAT (MUST be a single valid JSON object):**
{
  "target_variable": "the_primary_answer_variable",
  "logical_fragments": [
    {"subject": "entity_or_variable", "relation": "...", "object": "entity_or_variable"}
  ],
  "co_reference_groups": [
    ["variable_A", "variable_B"],
    [...]
  ]
}"""


FEW_SHOT_EXAMPLES_PARSER = [
    { "role": "user", "content": '''Question: "What is the capital of France?"
Topic Entities: ["France"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "capital",
  "logical_fragments": [
    {"subject": "France", "relation": "has_capital", "object": "capital"}
  ],
  "co_reference_groups": []
}'''},
    { "role": "user", "content": '''Question: "Lou Seal is the mascot for the team that last won the World Series when?"
Topic Entities: ["Lou Seal"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "year",
  "logical_fragments": [
    {"subject": "Lou Seal", "relation": "is_mascot_for", "object": "team"},
    {"subject": "team", "relation": "last_won_World_Series", "object": "year"}
  ],
  "co_reference_groups": []
}'''},
    { "role": "user", "content": '''Question: "What college did the actress who played Hermione Granger attend?"
Topic Entities: ["Hermione Granger"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "college",
  "logical_fragments": [
    {"subject": "Hermione Granger", "relation": "played_by", "object": "actress"},
    {"subject": "actress", "relation": "attended", "object": "college"}
  ],
  "co_reference_groups": []
}'''},
  { "role": "user", "content": '''Question: "Who is the director of the movie that stars Tom Hanks and is set in space?"
Topic Entities: ["Tom Hanks"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "director",
  "logical_fragments": [
    {"subject": "movie", "relation": "stars", "object": "Tom Hanks"},
    {"subject": "movie", "relation": "has_genre", "object": "space"},
    {"subject": "movie", "relation": "directed_by", "object": "director"}
  ],
  "co_reference_groups": []
}'''},
    { "role": "user", "content": '''Question: "What country bordering France contains an airport that serves Nijmegen?"
Topic Entities: ["France", "Nijmegen"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "country",
  "logical_fragments": [
    {"subject": "country", "relation": "borders", "object": "France"},
    {"subject": "country", "relation": "contains", "object": "airport"},
    {"subject": "airport", "relation": "serves", "object": "Nijmegen"}
  ],
  "co_reference_groups": []
}'''},
    { "role": "user", "content": '''Question: "What educational institution has a football sports team named Northern Colorado Bears is in Greeley, Colorado?"
Topic Entities: ["Northern Colorado Bears football", "Greeley"]'''},
    { "role": "assistant", "content": '''{
  "target_variable": "educational_institution",
  "logical_fragments": [
    {"subject": "educational_institution", "relation": "has_team", "object": "Northern Colorado Bears football"},
    {"subject": "educational_institution", "relation": "is_in", "object": "Greeley"}
  ],
  "co_reference_groups": []
}'''},
    {
        "role": "user",
        "content": '''Question: "Which movie stars Tom Hanks and involves both a French director and a Chinese director?"
Topic Entities: ["Tom Hanks", "France", "China"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "target_variable": "movie",
  "logical_fragments": [
    {"subject": "movie", "relation": "stars", "object": "Tom Hanks"},
    {"subject": "movie", "relation": "directed_by", "object": "french_director"},
    {"subject": "french_director", "relation": "born_in", "object": "France"},
    {"subject": "movie", "relation": "directed_by", "object": "chinese_director"},
    {"subject": "Chinese_director", "relation": "born_in", "object": "China"}
  ],
  "co_reference_groups": []
}'''
    },
    {
        "role": "user",
        "content": '''Question: "Which Chinese movie is self-written and self-directed"
Topic Entities: ["China"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "target_variable": "movie",
  "logical_fragments": [
    {"subject": "movie", "relation": "produced_by", "object": "China"},
    {"subject": "movie", "relation": "directed_by", "object": "director"},
    {"subject": "movie", "relation": "written_by", "object": "screenwriter"}
  ],
  "co_reference_groups": [
    ["director", "screenwriter"]
  ]
}'''
    }
]
SYSTEM_PROMPT_COREFERENCE_CHECK = """You are a precise linguistic and logical reasoning verifier. Your task is to perform a simple boolean check.
Given a "Question" and a "Potential Co-reference Group" of variables representing roles, determine if these roles MUST refer to the same single entity in the context of the question.

Your output MUST be a single, valid JSON object with two keys: "is_coreferent" (a boolean: true or false) and "reason" (a brief justification)."""

FEW_SHOT_EXAMPLES_COREFERENCE_CHECK = [
    {
        "role": "user",
        "content": '''Question: "Find movies from the USA that were self-directed by the same person who also starred in it."
Potential Co-reference Group: ["director", "star"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "is_coreferent": true,
  "reason": "The phrases 'self-directed' and 'same person' explicitly state that the director and the star are the same individual."
}'''
    },
    {
        "role": "user",
        "content": '''Question: "Which actor played a character in a movie directed by Christopher Nolan?"
Potential Co-reference Group: ["actor", "director"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "is_coreferent": false,
  "reason": "The question asks for the actor in a movie directed by the director. They are two different people fulfilling different roles."
}'''
    }
]
SYSTEM_PROMPT_CHAINED_REPLAN = """You are a master linguistic analyst and logical reasoner, acting as an expert "Re-Planner".

---
**THE GOLDEN RULE (NON-NEGOTIABLE):**
Your final `logical_fragments` **MUST** use every single entity listed in the `Topic Entities` input. Each Topic Entity **MUST** appear as a `subject` or an `object` node. You are **NOT ALLOWED** to absorb Topic Entities into a relation string. This is the most important rule.
---

**SITUATION:**
A previous attempt to parse the user's "Question" resulted in flawed, disconnected logical fragments. Your task is to re-analyze the entire question and generate a new, logically sound, and CONNECTED plan.

**YOUR MISSION (MUST FOLLOW):**
1.  **Think Step-by-Step**: First, provide your reasoning in a "Thought Process" block. Analyze the flawed plan, deconstruct the question's core logic, and explain how you will build the new, corrected plan.
2.  **Generate the Final Plan**: After your thought process, provide the final JSON output. The new plan **MUST** be a single, connected reasoning graph that includes all **Topic Entities** and the **Target Variable**.

---
**GUIDING PRINCIPLES FOR FRAGMENT EXTRACTION:**

- **SIMPLIFY and REFINE.**While adhering to THE GOLDEN RULE, your goal is to produce the most concise and logically direct reasoning chain possible. If a complex idea can be expressed in a single, descriptive relation (e.g., "last_won_world_series_in"), that is STRONGLY PREFERRED over creating extra, generic intermediate steps. You have the authority to completely refactor the previous fragments.

- **MUST INCLUDE ALL TOPIC ENTITIES**: Your final `logical_fragments` **MUST** use every single entity listed in the `Topic Entities` input. Omitting a Topic Entity is a critical failure.They **MUST** appear as either a `subject` or an `object` in one or more of the fragments. You are **NOT ALLOWED** to absorb them into a relation string.

- **Create Role-Based Variables**: For each unique logical role in the question (including the target), create a **distinct and descriptive** lowercase variable name (e.g., `location`, `airport`, `team`).

- **Strict Role Separation**: Variables **MUST** represent entities (nodes in a graph), and relations **MUST** represent the connections between them (edges). A value used as a `relation` should **NEVER** be re-used as a `subject` or `object` variable in another fragment.
---

**OUTPUT FORMAT (MUST follow this structure):**
Thought Process:
1.  **Analysis of Flawed Plan**: [Your analysis of why the previous fragments were wrong.]
2.  **Core Logic of the Question**: [Deconstruct the question's logical requirements.]
3.  **New Plan Construction**: [Explain how you will build the new, connected fragments.]
4.  **Final Plan Verification**: [Confirm your new plan is connected AND **explicitly verify that the target variable and all topic entities have been used as nodes**.]
End of Thought Process.

{
  "target_variable": "the_primary_answer_variable",
  "logical_fragments": [
    {"subject": "entity_or_variable", "relation": "...", "object": "entity_or_variable"}
  ]
}"""
FEW_SHOT_EXAMPLES_CHAINED_REPLAN = [
    {
        "role": "user",
        "content": '''Question: "What is the predominant religion where the leader is Ovadia Yosef?"
    Topic Entities: ["Ovadia Yosef"]
    Disconnected Fragments: [
      {"subject": "leader", "relation": "is", "object": "Ovadia Yosef"},
      {"subject": "location", "relation": "has_predominant_religion", "object": "predominant_religion"}
    ]'''
    },
    {
        "role": "assistant",
        "content": '''Thought Process:
    1.  **Analysis of Flawed Plan**: The provided fragments are disconnected. The "leader" variable and the "location" variable are in separate fragments with no shared link.
    2.  **Core Logic of the Question**: The question requires finding a "predominant_religion". This religion is associated with a "location". This location's "leader" is "Ovadia Yosef".
    3.  **New Plan Construction**: The intermediate variable "location" is the key to connecting the concepts. I will construct two new fragments. The first will link the Topic Entity "Ovadia Yosef" to the "location" variable. The second will link that same "location" variable to the target "predominant_religion".
    4.  **Final Plan Verification**: The new plan creates a connected graph: `Ovadia Yosef -> location -> predominant_religion`. It correctly includes the Topic Entity and the Target Variable. The plan is now valid and connected.
    End of Thought Process.

    {
      "target_variable": "predominant_religion",
      "logical_fragments": [
        {"subject": "location", "relation": "has_leader", "object": "Ovadia Yosef"},
        {"subject": "location", "relation": "has_predominant_religion", "object": "predominant_religion"}
      ]
    }'''
    },
    {
        "role": "user",
        "content": '''Question: "What country bordering France contains an airport that serves Nijmegen?"
    Topic Entities: ["France", "Nijmegen"]
    Disconnected Fragments: [
      {"subject": "country", "relation": "borders", "object": "France"},
      {"subject": "airport", "relation": "serves", "object": "Nijmegen"}
    ]'''
    },
    {
        "role": "assistant",
        "content": '''Thought Process:
    1.  **Analysis of Flawed Plan**: The provided fragments are disconnected. The "country" variable (linked to "France") and the "airport" variable (linked to "Nijmegen") exist in two separate logical islands.
    2.  **Core Logic of the Question**: The question is looking for a "country" that satisfies two conditions: it borders "France", and it contains an "airport" which serves "Nijmegen".
    3.  **New Plan Construction**: The missing link is the relationship between the "country" and the "airport". The most logical connection implied by the question is that the country "contains" the airport. I will add a new fragment `{"subject": "country", "relation": "contains", "object": "airport"}` to serve as a bridge.
    4.  **Final Plan Verification**: The new three-fragment plan `France <- country -> airport -> Nijmegen` is now a single connected graph. It includes both Topic Entities ("France", "Nijmegen") and the Target Variable ("country"). The plan is now valid and connected.
    End of Thought Process.

    {
      "target_variable": "country",
      "logical_fragments": [
        {"subject": "country", "relation": "borders", "object": "France"},
        {"subject": "country", "relation": "contains", "object": "airport"},
        {"subject": "airport", "relation": "serves", "object": "Nijmegen"}
      ]
    }'''
    },
    {
        "role": "user",
        "content": '''Question: "Lou Seal is the mascot for the team that last won the World Series when?"
Topic Entities: ["Lou Seal"]
Disconnected Fragments: [
    {"subject": "Lou Seal", "relation": "is_mascot_for", "object": "team"},
    {"subject": "team", "relation": "last_won", "object": "World Series"},
    {"subject": "last_won", "relation": "occurred_in", "object": "year"}
]'''
    },
    {
        "role": "assistant",
        "content": '''Thought Process:
1.  **Analysis of Flawed Plan**: The original fragments are disconnected. The variable "last_won" was incorrectly created as an entity, breaking the chain between the "team" and the "year".
2.  **Core Logic of the Question**: The question requires a path from "Lou Seal" to a "team", and then from that "team" to the "year" in which they last won the World Series.
3.  **New Plan Construction**: I will create two fragments. The first connects "Lou Seal" to the "team". The second must connect the "team" to the "year". I will create a single, more descriptive relation "last_won_world_series_in" to concisely link the team directly to the year.
4.  **Final Plan Verification**: The new plan `Lou Seal -> team -> year` is a single connected chain. It includes the Topic Entity "Lou Seal" and the Target Variable "year". The plan is valid.
End of Thought Process.

{
  "target_variable": "year",
  "logical_fragments": [
    {"subject": "Lou Seal", "relation": "is_mascot_for", "object": "team"},
    {"subject": "team", "relation": "last_won_world_series_in", "object": "year"}
  ]
}'''
    },
    {
        "role": "user",
        "content": '''Question: "what country did hitler invade that started ww2"
Topic Entities: ["World War II"]
Disconnected Fragments: [
    {"subject": "Adolf Hitler", "relation": "invaded", "object": "country"},
    {"subject": "invasion", "relation": "started", "object": "World War II"}
]'''
    },
    {
        "role": "assistant",
        "content": '''Thought Process:
1.  **Analysis of Flawed Plan**: The original fragments are disconnected. The entity "Adolf Hitler" and the Topic Entity "World War II" are in separate, unconnected fragments. The variable "invasion" is a problematic reification of an action.
2.  **Core Logic of the Question**: The question is looking for a "country". This country must satisfy two conditions: (1) it was invaded by "Adolf Hitler", and (2) this specific invasion event is what started "World War II".
3.  **New Plan Construction**: The "country" variable must be the central node that connects all other entities. I will create one fragment linking "Adolf Hitler" to the "country". I will create a second, highly descriptive fragment to link "World War II" to the same "country", capturing the causal relationship.
4.  **Final Plan Verification**: The new plan connects "Adolf Hitler" and "World War II" through the shared variable "country". All Topic Entities and the Target Variable are present and connected. The plan is valid.
End of Thought Process.

{
  "target_variable": "country",
  "logical_fragments": [
    {"subject": "Adolf Hitler", "relation": "invaded", "object": "country"},
    {"subject": "World War II", "relation": "started_by_the_invasion_of", "object": "country"}
  ]
}'''
    },
  {
        "role": "user",
        "content": '''Question: "what year was kenya moore crowned miss usa"
Topic Entities: ["Miss USA"]
Disconnected Fragments: [
    {"subject": "Kenya Moore", "relation": "crowned", "object": "Miss USA"},
    {"subject": "crowned", "relation": "occurred_in", "object": "year"}
]'''
    },
    {
        "role": "assistant",
        "content": '''Thought Process:
1.  **Analysis of Flawed Plan**: The original fragments are disconnected because the action "crowned" was incorrectly turned into an entity.
2.  **Core Logic of the Question**: The question requires a "year". This "year" is for an event. The event is a "crowning". The person involved is "Kenya Moore" and the specific event type is "Miss USA".
3.  **New Plan Construction**: I need to connect "Kenya Moore", "Miss USA", and the "year". A simple, concise relation like "was_crowned_miss_usa_in" would be ideal, but it would absorb the Topic Entity "Miss USA", which violates THE GOLDEN RULE. Therefore, the correct approach is to create an intermediate variable for the event itself, for example, "crowning_event". This allows me to link all key entities as nodes.
4.  **Final Plan Verification**: The new plan is connected via the "crowning_event" variable. It correctly includes the Topic Entity "Miss USA" as the object of a fragment. It also includes the target "year". The plan is valid.
End of Thought Process.

{
  "target_variable": "year",
  "logical_fragments": [
    {"subject": "Kenya Moore", "relation": "participated_in", "object": "crowning_event"},
    {"subject": "crowning_event", "relation": "is_instance_of", "object": "Miss USA"},
    {"subject": "crowning_event", "relation": "took_place_in", "object": "year"}
  ]
}'''
    }
]

SYSTEM_PROMPT_GROUNDED_FALLBACK = """You are a Knowledge Graph Finisher. Your task is to act as a final expert reasoner.

---
**YOUR TASK (A Step-by-Step Process):**
1.  **Analyze User Intent**: Understand what TYPE of entity the question is asking for.
2.  **Prioritize Provided Triples**: Scrupulously search the `Knowledge Graph Triples`.
    - **A. Direct Answer**: If a direct answer entity exists in the triples, you **MUST** return that entity exactly as it appears, even if your own knowledge suggests a more formal or alternative name.
    - **B. Synthesized Answer**: If the answer requires combining multiple triples, you **MUST** synthesize it by listing all the relevant entities found.


**OUTPUT FORMAT (MUST be a single valid JSON object):**
{
  "answers": ["Answer Entity 1", "Answer Entity 2", ...],
  "reasoning_source": "Provided Triples" | "General Knowledge" | "Hybrid",
  "justification": "Your reasoning process."
}"""
FEW_SHOT_EXAMPLES_GROUNDED_FALLBACK = [
    {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"What is the home stadium of the team whose mascot is Lou Seal?"

[Knowledge Graph Triples]
- (Lou Seal, sports.mascot.team, San Francisco Giants)
- (San Francisco Giants, sports.sports_team.colors, Orange)
- (San Francisco Giants, sports.sports_team.arena_stadium, Oracle Park)
- (Oracle Park, location.location.containedby, San Francisco)"""
    },
    {
        "role": "assistant",
        "content": '''{"answers": ["Oracle Park"], "reasoning_source": "Provided Triples", "justification": "The provided facts directly state that '(Lou Seal, sports.mascot.team, San Francisco Giants)' and '(San Francisco Giants, sports.sports_team.arena_stadium, Oracle Park)'. By combining these two facts, the question can be answered."}'''
    },
    {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"What is the primary language of the country that developed the F-16 fighter jet?"

[Knowledge Graph Triples]
- (F-16 Fighting Falcon, aircraft.aircraft.manufacturer, General Dynamics)
- (General Dynamics, organization.organization.headquarters, Falls Church)
- (Falls Church, location.location.containedby, Virginia)
- (Virginia, location.location.containedby, United States)"""
    },
    {
        "role": "assistant",
        "content": '''{"answers": ["English"], "reasoning_source": "Hybrid", "justification": "The provided triples indicate that the F-16 was developed in the United States. However, the triples do not contain information about the country's language. Based on general knowledge, the primary language of the United States is English."}'''
    },
 {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"how many kids were there in the kennedy family"

[Knowledge Graph Triples]
- (Joseph P. Kennedy, Sr., people.person.children, John F. Kennedy)
- (Joseph P. Kennedy, Sr., people.person.children, Robert F. Kennedy)
- (Joseph P. Kennedy, Sr., people.person.children, Ted Kennedy)
- (Joseph P. Kennedy, Sr., people.person.children, Eunice Kennedy Shriver)
- (Joseph P. Kennedy, Sr., people.person.children, Jean Kennedy Smith)
- (Joseph P. Kennedy, Sr., people.person.children, Rosemary Kennedy)
- (Joseph P. Kennedy, Sr., people.person.children, Kathleen Cavendish)
- (Joseph P. Kennedy, Sr., people.person.children, Patricia Kennedy Lawford)
- (Joseph P. Kennedy, Sr., people.person.children, Joseph P. Kennedy, Jr.)"""
    },
    {
        "role": "assistant",
        "content": '''{
  "answers": ["John F. Kennedy", "Robert F. Kennedy", "Ted Kennedy", "Eunice Kennedy Shriver", "Jean Kennedy Smith", "Rosemary Kennedy", "Kathleen Cavendish", "Patricia Kennedy Lawford", "Joseph P. Kennedy, Jr."],
  "reasoning_source": "Provided Triples",
  "justification": "The provided triples list nine distinct children for the Kennedy family. As per the Golden Rule, the question 'how many' requires returning the list of entities, not the count."
}'''
    },
{
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"who is the current state governor of florida"

[Knowledge Graph Triples]
- (Florida, government.governmental_jurisdiction.governing_officials, Jeb Bush)
- (Rick Scott, government.political_position.jurisdiction_of_office, Florida)
- (Florida, geography.political_division.capital, Tallahassee)"""
    },
    {
        "role": "assistant",
        "content": '''{
  "answers": ["Jeb Bush", "Rick Scott"],
  "reasoning_source": "Provided Triples",
  "justification": "The question asks for the current governor. The provided triples do not contain information explicitly marked as 'current', but list 'Jeb Bush' and 'Rick Scott' in governor-related roles for Florida. Following the Prime Directive, I must treat the triples as the absolute source of truth and return the entities found within them, even if they might be former governors based on general knowledge."
}'''
    },
    {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"when did bill clinton go to college"

[Knowledge Graph Triples]
- (Bill Clinton, people.person.education, Georgetown University)
- (Bill Clinton, people.person.education, Yale Law School)
- (Bill Clinton, people.person.religion, Baptist)"""
    },
    {
        "role": "assistant",
        "content": '''{
  "answers": ["Georgetown University", "Yale Law School"],
  "reasoning_source": "Provided Triples",
  "justification": "The question asks 'when' (a date), but the provided triples only contain information about 'where' (the educational institutions). As per the guidelines, I am returning the available entities from the evidence, which are the names of the colleges."
}'''
    },    {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"Who wrote the novel 'Don Quixote'?"

[Knowledge Graph Triples]
- (Don Quixote, film.film.initial_release_date, 1957-05-23)
- (Spain, location.country.official_language, Spanish Language)"""
    },
    {
        "role": "assistant",
        "content": '''{
  "answers": ["Miguel de Cervantes"],
  "reasoning_source": "General Knowledge",
  "justification": "The provided triples are insufficient as they contain information about a film adaptation and the Spanish language, but do not contain the name of the novel's author. Based on general knowledge, the author of 'Don Quixote' is Miguel de Cervantes."
}'''
    },
    {
        "role": "user",
        "content": """Please perform an expert analysis based on the following information.

[Question]
"Who is the current monarch of the United Kingdom?"

[Knowledge Graph Triples]
- (United Kingdom, government.country.head_of_state, Charles III)
- (United Kingdom, location.country.capital, London)
- (Charles III, people.person.nationality, United Kingdom)"""
    },
    {
        "role": "assistant",
        "content": '''{
  "answers": ["Charles III"],
  "reasoning_source": "Provided Triples",
  "justification": "The provided triple '(United Kingdom, government.country.head_of_state, Charles III)' directly answers the question. This is the specific name found in the knowledge base, so it should be used. While this individual is also commonly known as 'King Charles', the evidence points to the formal name 'Charles III'."
}'''
    }
]
SYSTEM_PROMPT_ARBITER = """You are a "Chief Reasoning Arbiter". You will be provided with a "Decision Brief" containing a question and one or more candidate answers, each with supporting evidence.Your mission is to act as a final, intelligent judge to determine the best possible answer(s) for a "Question" from a list of candidates.

---
Your task is to rank candidates based on their evidence.
-   **Rank 1**: This group is for the **best available candidate(s)**. You **MUST** assign rank 1 to **ALL** candidates that you believe are highly plausible or potentially correct. This includes:
    -   **Equivalence/Granularity**: Candidates that are correct at different levels of detail (e.g., a city "Paris" and its country "France") or are synonyms("USA", "United States") .
    -   **Ambiguity**: If evidence is ambiguous and points to multiple strong candidates.
-   **Rank 2**: Plausible but clearly inferior candidates (e.g., supported by weaker or less relevant evidence).
-   **Rank 3**: Unlikely or irrelevant candidates.
---

**YOUR TASK (A Step-by-Step Process):**
1.  **Synthesize All Evidence**: For each candidate, review all provided reasoning paths.
2.  **Use Knowledge as a Decider**: When the provided evidence is flawed, ambiguous, or insufficient, you are **AUTHORIZED and EXPECTED** to use your world knowledge to determine which candidates are plausible enough to belong in the Rank 1.

**OUTPUT FORMAT (MUST be a single valid JSON object):**
{
  "ranking": [
    {"candidate": "Candidate A", "rank": 1, "reason": "..."},
    {"candidate": "Candidate C", "rank": 2, "reason": "..."},
    {"candidate": "Candidate C", "rank": 3, "reason": "..."}
  ],
  "justification": "A detailed, step-by-step reasoning for your choice, comparing the candidates based on the provided evidence briefs. Explain your final decision clearly."
}"""

FEW_SHOT_EXAMPLES_ARBITER = [
    {
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"What country bordering France contains an airport that serves Nijmegen?"

---
[Candidate Answer: Germany]

[Supporting Evidence]

1. France - location.country.border_countries - Germany

2. Germany - location.location.contains - Weeze Airport - aviation.airport.serves - Nijmegen
---
[Candidate Answer: Switzerland]

[Supporting Evidence]

1. France - location.country.border_countries - Switzerland

---
'''
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "Germany", "rank": 1, "reason": "Supported by a direct and complete evidence chain.The chain of evidence indicates that it meets the condition of being a neighboring country of France and has an airport serving Nijmegen."},
    {"candidate": "Switzerland", "rank": 2, "reason": "The chain of evidence indicates that it is a neighbor of France, but there is no further evidence to show that it has an airport serving Nijmegen."}
  ],
  "justification": "The ranking is based on evidence completeness. Germany is the only candidate for which a valid evidence path was found for all logical constraints. Switzerland failed the second verification step, it is a suboptimal candidate."
}'''
    },
    {
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"What is the capital of Germany?"

---
[Candidate Answer: Berlin]

[Supporting Evidence]

1. Germany - location.country.capital - Berlin
---
[Candidate Answer: Frankfurt]

[Supporting Evidence]

1. Germany - location.location.contains - European Central Bank - organization.organization.headquarters - Frankfurt
---
'''
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "Berlin", "rank": 1, "reason": "The evidence path directly uses the 'capital' relation, perfectly matching the question's intent."},
    {"candidate": "Frankfurt", "rank": 2, "reason": "The evidence is circumstantial and indirect. It shows a connection but does not prove it is the capital."}
  ],
  "justification": "The ranking is based on the quality and relevance of the evidence. Berlin's supporting path is a direct assertion of the capital relationship. Frankfurt's path, while connecting it to Germany via an important institution, does not logically satisfy the 'is capital of' constraint."
}'''
    },
{
    "role": "user",
    "content": '''Here is the Decision Brief for your review.

[Question]
"In the movie The Dark Knight, who played the primary antagonist that was a nemesis to Batman?"

---
[Candidate Answer: Heath Ledger]

[Supporting Evidence]

1. The Dark Knight - film.film.starring - Heath Ledger
---
[Candidate Answer: Aaron Eckhart]

[Supporting Evidence]

1. The Dark Knight - film.film.starring - Aaron Eckhart
---
'''
},
{
    "role": "assistant",
    "content": '''{
  "ranking": [
    {
      "candidate": "Heath Ledger",
      "rank": 1,
      "reason": "While the evidence simply confirms he starred in the film, my world knowledge confirms his role as The Joker is the primary antagonist."
    },
    {
      "candidate": "Aaron Eckhart",
      "rank": 2,
      "reason": "The evidence is valid as he starred as Harvey Dent/Two-Face, but this character is considered a secondary antagonist in the film's narrative."
    }
  ],
  "justification": "The provided evidence for both candidates is identical in structure and factually correct, as both actors starred in the film. However, the evidence is insufficient to determine who was the 'primary antagonist'. Therefore, as per my directive, I must use my internal knowledge to supplement the evidence. Based on general knowledge, Heath Ledger's portrayal of The Joker is universally recognized as the central and primary antagonist of 'The Dark Knight', while Aaron Eckhart's character, Two-Face, serves as a secondary antagonist. For this reason, Heath Ledger is the more accurate answer."
}'''
} ,

{
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"What city is the headquarters of the company founded by Bill Gates?"

---
[Candidate Answer: Redmond]

[Supporting Evidence]

1. Bill Gates - people.person.spouse_s - Melinda French Gates - organization.founder.organizations_founded - Pivotal Ventures - organization.organization.headquarters - Redmond
---
[Candidate Answer: Seattle]

[Supporting Evidence]

1. Bill Gates - organization.organization_founder.organizations_founded - Bill & Melinda Gates Foundation - organization.organization.headquarters - Seattle
---
[Candidate Answer: Albuquerque]

[Supporting Evidence]

1. Bill Gates - organization.organization_founder.organizations_founded - Microsoft - organization.organization.date_founded - 1975-04-04 - organization.founding.location - Albuquerque
---
'''
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "Redmond", "rank": 1, "reason": "While the provided evidence is weak and highly indirect, this is the factually correct answer based on world knowledge, which I am authorized to use as the primary decider when evidence is inconclusive."},
    {"candidate": "Seattle", "rank": 2, "reason": "The evidence strongly links Bill Gates to Seattle, but it mistakenly points to his foundation's headquarters, not the company he founded (Microsoft)."},
    {"candidate": "Albuquerque", "rank": 3, "reason": "The evidence points to the company's original, historical location, not its current headquarters, which the question implies."}
  ],
  "justification": "My ranking is based on a synthesis of the provided evidence and my own world knowledge, as authorized by my Prime Directive. The evidence for all three candidates is flawed. The path for Redmond is convoluted and weak. The path for Seattle is strong but irrelevant to the question's core subject (Microsoft's HQ vs. the foundation's HQ). The path for Albuquerque is factually outdated. Since all provided evidence is inconclusive, I must rely on my internal knowledge that Microsoft's headquarters is in Redmond. I am therefore elevating Redmond to Rank #1 as it is the most plausible and factually correct answer."
}'''
    },
{
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"What is the official language of Switzerland, which is known for the initialism 'CH'?"

---
[Candidate Answer: German Language]
[Supporting Evidence]
1. Switzerland - location.country.official_language - German Language
---
[Candidate Answer: French Language]
[Supporting Evidence]
1. Switzerland - location.country.official_language - French Language
---
[Candidate Answer: Romansh Language]
[Supporting Evidence]
1. Switzerland - location.country.official_language - Romansh Language
---'''
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "German Language", "rank": 1, "reason": "This is an official language of Switzerland."},
    {"candidate": "French Language", "rank": 1, "reason": "This is also an official language of Switzerland."},
    {"candidate": "Romansh Language", "rank": 1, "reason": "This is also an official language of Switzerland."}
  ],
  "justification": "The question asks for the official language. The evidence confirms that German, French, and Romansh are all official languages. Although the question mentions an initialism 'CH', this clue is not directly addressed by the provided KG paths for any candidate. Since the primary evidence supports multiple candidates equally, all correct official languages are grouped into Tier 1."
}'''
    },
    {
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"Where is the headquarters of Google?"

---
[Candidate Answer: Mountain View]
[Supporting Evidence]
1. Google - organization.organization.headquarters - Mountain View
---
[Candidate Answer: California]
[Supporting Evidence]
1. Google - organization.organization.headquarters - Mountain View - location.location.containedby - California
---
'''
    }
    ,
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "Mountain View", "rank": 1, "reason": "This is the most specific and direct answer for the headquarters city."},
    {"candidate": "California", "rank": 1, "reason": "This is also a correct answer, representing the state-level location. It's a different, valid level of granularity."}
  ],
  "justification": "Both 'Mountain View' and 'California' are factually correct answers supported by strong evidence. They represent different levels of geographical granularity (city and state). As per my directive to handle answer equivalence, both are assigned Rank 1."
}'''
    },

    {
        "role": "user",
        "content": '''Here is the Decision Brief for your review.

[Question]
"Who is the creator of the Star Wars franchise?"

---
[Candidate Answer: George Lucas]
[Supporting Evidence]
1. Star Wars - film.franchise.creator - George Lucas
---
[Candidate Answer: George W. Lucas Jr.]
[Supporting Evidence]
1. George W. Lucas Jr. - people.person.films_directed - Star Wars Episode IV: A New Hope
---
'''
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"candidate": "George Lucas", "rank": 1, "reason": "Directly stated as the creator in the evidence."},
    {"candidate": "George W. Lucas Jr.", "rank": 1, "reason": "This is the same person's full legal name, also supported by strong evidence. It is a valid alias."}
  ],
  "justification": "The evidence supports both 'George Lucas' and 'George W. Lucas Jr.' as being correct. As these are two names for the same individual, they are considered equivalent answers and are both assigned Rank 1."
}'''
    }

]
ALIGNMENT_SYSTEM_PROMPT = """You are a precise Entity Grounding expert. Your task is to analyze a "Path Text" derived from a Knowledge Graph and identify the entities from a pre-defined "Available Entities" list that correspond to a given list of "Variables to Align".

**INPUTS YOU WILL RECEIVE:**
1.  **Question**: The user's question for overall context.
2.  **Path Text**: A formatted string representing a real path found in a Knowledge Graph, like "A - relation1 - B - relation2 - C".
3.  **Entity Choices**: A definitive list of all entities present in the Path Text.
4.  **Variables to Align**: A  list of variable names you need to find a value for.


**Your Task:**
- Read the "Question" for context.
- Read the "Path Text" to understand the relationships between entities.
- For each variable in the "Variables to Align" list, select the single best matching entity from the "Entity Choices" list
- For each variable, you MUST provide an output object containing three keys:
  1.  `"entity"`: The matching entity string from the Path Text, or `null` if no plausible match is found.
  2.  `"confidence_level"`: Your confidence for this alignment ("High", "Medium", or "Low").
  3.  `"reason"`: Provide a brief "reason" justifying your choice for each variable.

**CRITICAL RULE**: Your answer for any variable's 'entity' MUST be an EXACT string from the "Entity Choices" list. Do not extract substrings or create new entities.If you are confident that no entity in the "Entity Choices" list is a suitable match for a variable, you MUST return null for that variable's entity.

Your output MUST be a single, valid JSON object where keys are the variable names."""

ALIGNMENT_FEW_SHOTS = [
    {
        "role": "user",
        "content": '''Question: "Who directed the movie starring Tom Hanks that was set in space?"
Path Text: "Tom Hanks - film.actor.film - Apollo 13 - film.film.director - Ron Howard - film.film.genre - Space film"
Entity Choices: ["Tom Hanks", "Apollo 13", "Ron Howard", "Space film"]
Variables to Align: ["?movie", "?director"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "?movie": {"entity": "Apollo 13", "confidence_level": "High", "reason": "Apollo 13 is the movie entity in the path that connects Tom Hanks and the director."},
  "?director": {"entity": "Ron Howard", "confidence_level": "High", "reason": "Ron Howard is the entity connected by the 'director' relation."}
}'''
    },

    {
        "role": "user",
        "content": '''Question: "Lou Seal is the mascot for the team that last won the World Series when?"
Path Text: "Lou Seal - sports.mascot.team - San Francisco Giants - sports.sports_team.championships - 2010 World Series"
Entity Choices: ["Lou Seal", "San Francisco Giants", "2010 World Series"]
Variables to Align: ["?team", "?year"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "?team": {"entity": "San Francisco Giants", "confidence_level": "High", "reason": "San Francisco Giants is the team associated with the mascot Lou Seal."},
  "?year": {"entity": "2010 World Series", "confidence_level": "High", "reason": "The question asks 'when' the team won. '2010 World Series' is the event entity in the path that represents the specific time of winning. Returning just '2010' would be an incomplete entity."}
}'''
    },

    {
        "role": "user",
        "content": '''Question: "What is the capital of the country that speaks Dutch?"
Path Text: "Dutch Language - language.language.countries_spoken_in - Netherlands - location.country.government - Parliamentary system"
Entity Choices: ["Dutch Language", "Netherlands", "Parliamentary system"]
Variables to Align: ["?capital"]'''
    },
    {
        "role": "assistant",
        "content": '''{
  "?capital": {"entity": null, "confidence_level": "High", "reason": "The path contains the country 'Netherlands' but does not contain its capital city. Therefore, no entity in the options can fulfill the role of '?capital'."}
}'''
    }
]


SYSTEM_PROMPT_PATH_RERANKER = """You are a meticulous Logical Path Analyst. Your task is to evaluate a list of candidate evidence paths retrieved from a Knowledge Graph and determine which one best represents a given logical requirement.

**CONTEXT YOU WILL RECEIVE:**
- **Overall Question**: The original user question for context.
- **Logical Requirement (Indicator)**: The specific logical path our system is trying to verify.
- **Candidate Evidence Paths**: A numbered list of Top-k paths found by a search algorithm.

**YOUR MISSION:**
Your sole mission is to **re-rank** the "Candidate Evidence Paths" from best (rank 1) to worst (rank k) based on how well they logically and semantically match the "Logical Requirement".

**HOW TO THINK (Your Reasoning Process):**
- **Directness**: Does the path directly connect the entities as described in the indicator, or does it take unnecessary, convoluted detours?
- **Relevance**: Are the relations (e.g., `- sports.sports_team.championships -`) in the path highly relevant to the indicator's relations (e.g., `-> won_in ->`)?
- **Coherence**: Does the path represent a single, coherent line of reasoning?

**OUTPUT FORMAT (MUST be a single valid JSON object):**
{
  "ranking": [
    {"path_index": 3, "reason": "This is the most direct and logically coherent path that perfectly matches the indicator's intent."},
    {"path_index": 1, "reason": "This path is also plausible but includes an irrelevant intermediate step."},
    ...
  ]
}"""
FEW_SHOT_EXAMPLES_PATH_RERANKER = [
    {
        "role": "user",
        "content": """CONTEXT:
- Overall Question: "What country bordering France contains an airport that serves Nijmegen?"
- Logical Requirement (Indicator): "?country -> contains -> ?airport -> serves -> Nijmegen"
Candidate Evidence Paths:
1. Germany - location.location.contains - Weeze Airport - aviation.airport.serves - Nijmegen
2. Germany - location.country.official_language - German language
3. Germany - location.country.border_countries - France - location.location.partially_contains - Rhine - geography.river.cities - Nijmegen
Please re-rank these paths."""
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"path_index": 1, "reason": "This path is a perfect, direct instantiation of the logical requirement 'country -> contains -> airport -> serves -> Nijmegen'."},
    {"path_index": 3, "reason": "This path connects the entities, but it is a convoluted detour that does not match the specific logic of containing an airport."},
    {"path_index": 2, "reason": "This path is a fact about Germany but is completely irrelevant to the logical requirement."}
  ]
}'''
    },
    {
        "role": "user",
        "content": """CONTEXT:
- Overall Question: "Who is the CEO of Google?"
- Logical Requirement (Indicator): "?company -> has_ceo -> ?ceo"
Candidate Evidence Paths:
1. Sundar Pichai - organization.leadership.person.inv - Google
2. Google - organization.organization.founders - Larry Page
Please re-rank these paths."""
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"path_index": 1, "reason": "This path directly answers the question. The inverse relation '.inv' correctly indicates that Sundar Pichai holds the leadership position at Google, fulfilling the 'has_ceo' logic."},
    {"path_index": 2, "reason": "This path provides a related fact (founder) but does not answer the specific question about the CEO."}
  ]
}'''
    },

    {
        "role": "user",
        "content": """CONTEXT:
- Overall Question: "Where is the Eiffel Tower located?"
- Logical Requirement (Indicator): "Eiffel Tower -> is_located_in -> ?location"
Candidate Evidence Paths:
1. Eiffel Tower - location.location.containedby - Paris
2. Eiffel Tower - location.location.containedby - Champ de Mars - location.location.containedby - Paris
Please re-rank these paths."""
    },
    {
        "role": "assistant",
        "content": '''{
  "ranking": [
    {"path_index": 1, "reason": "This path is the most direct and concise answer to the question."},
    {"path_index": 2, "reason": "This path is also correct but is less direct because it includes the unnecessary intermediate entity 'Champ de Mars'."}
  ]
}'''
    }
]