Data Types
================
L2P serves as a intermediary between Large Language Models (LLMs) and the construction of reliable planning components in PDDL. To ensure consistency and reliability in planning workflows, L2P translates LLM-generated outputs into well-defined Python types. These types form the foundation for creating, manipulating, and validating planning components, enabling easy integration of LLM capabilities in PDDL. You can find more information on PDDL `here <https://planning.wiki/guide/whatis/pddl>`_.

Currently, L2P is currently at work to integrate more advanced PDDL aspects. Below you will find the types and respective Python format:

Types
-------------------------------------------------------
**Types** are formatted as a Python dictionary: `dict[str,str]` or a nested list dictionary `list[dict[str,str]]` where the first string represents the PDDL type name, and the second string provides a natural language description of the type.

**Example One (Basic types)**: ::
    
    {
        "type_1": "description",
        "type_2": "description",
        "type_3": "description",
    }

Converts to this string using `format_types_to_string(types)` ::

    type_1 ; description
    type_2 ; description
    type_3 ; description

**Example Two (Nested types)**: ::
    
    [
        {
            "parent_type_1": "description for parent type 1",
            "children": [
                {
                    "child_type_1": "description for child type 1",
                    "children": [
                        {"child_child_type_1": "description for child child type 1", "children": []},
                        {"child_child_type_2": "description for child child type 1", "children": []}
                    ]
                }
            ]
        },
        {
            "parent_type_2": "description for parent type 2",
            "children": [
                {
                    "child_type_2": "description for child type 2",
                    "children": [
                        {"child_child_type_3": "description for child child type 3", "children": []}
                    ]
                }
            ]
        }
    ]

Converts to this string using `format_types_to_string(types)` ::

    parent_type_1 ; description for parent type 1
    child_type_1 - parent_type_1 ; description for child type 1
    child_child_type_1 - child_type_1 ; description for child child type 1
    child_child_type_2 - child_type_1 ; description for child child type 1
    parent_type_2 ; description for parent type 2
    child_type_2 - parent_type_2 ; description for child type 2
    child_child_type_3 - child_type_2 ; description for child child type 3


Predicates
-------------------------------------------------------
A **Predicate** is a class defined as a `TypeDict` in Python. Each key specifies a property of the predicate:

- **name** (*str*): The name of the predicate.
- **desc** (*Optional[str]*): An optional natural language description of the predicate.
- **raw** (*str*): The raw representation of the predicate as defined in PDDL.
- **params** (*ParameterList*): A list of parameters associated with the predicate.
- **clean** (*str*): A cleaned or standardized version of the predicate for easier processing or display.

Where **ParameterList** is: ::

    ParameterList = NewType("ParameterList", OrderedDict[str, str])  # {'?param_name': 'param_type'} OR OrderedDict([('?param_name', 'param_type')])

For example, **formalize_predicates()** takes the LLM output: ::

    ### New Predicates
    ```
    - (predicate_name_1 ?t1 - type_1 ?t2 - type_2): 'predicate_description'
    - (predicate_name_2 ?t3 - type_3 ?t4 - type_4): 'predicate_description'
    - (predicate_name_3 ?t5 - type_5): 'predicate_description'
    ``` 

And converts it into this: ::

    [
        {
            'name': 'predicate_name_1', 
            'desc': "'predicate_description'", 
            'raw': "(predicate_name_1 ?t1 - type_1 ?t2 - type_2): 'predicate_description'", 
            'params': OrderedDict({'?t1': 'type_1', '?t2': 'type_'}), 
            'clean': '(predicate_name_1 ?t1 - type_1 ?t2 - type_)'
        }, 
        {
            'name': 'predicate_name_2', 
            'desc': "'predicate_description'", 
            'raw': "(predicate_name_2 ?t3 - type_3 ?t4 - type_4): 'predicate_description'", 
            'params': OrderedDict({'?t3': 'type_3', '?t4': 'type_'}), 
            'clean': '(predicate_name_2 ?t3 - type_3 ?t4 - type_)'
        }, 
        {
            'name': 'predicate_name_3', 
            'desc': "'predicate_description'", 
            'raw': "(predicate_name_3 ?t5 - type_5): 'predicate_description'", 
            'params': OrderedDict({'?t5': 'type_'}), 
            'clean': '(predicate_name_3 ?t5 - type_)'
        }
    ]

**NOTE:** `:functions` type class follows the same format as `:predicates` in L2P.

Action
-------------------------------------------------------
An **Action** is a class defined as a `TypeDict` in Python. Each key specifies a property of the action schema:

- **name** (*str*): The name of the action
- **raw** (*str*): The raw representation of the action as defined in PDDL.
- **params** (*ParameterList*): A list of parameters associated with the action.
- **preconditions** (*str*): 
- **effects** (*str*):

For example, **formalize_pddl_action()** takes the LLM output: ::

    ### Action Parameters
    ```
    - ?b1 - block: The block being stacked on top
    - ?b2 - block: The block being stacked upon
    - ?a - arm: The arm performing the stacking action
    ```

    ### Action Preconditions
    ```
    (and
        (holding ?a ?b1) ; The arm is holding the top block
        (clear ?b2) ; The bottom block is clear
    )
    ```

    ### Action Effects
    ```
    (and
        (not (holding ?a ?b1)) ; The arm is no longer holding the top block
        (on ?b1 ?b2) ; The top block is now on the bottom block
        (not (clear ?b2)) ; The bottom block is no longer clear
    )
    ```

And converts it into this: ::

    action: Action = {'name': 'stack', 
                    'params': OrderedDict([('?b1', 'block'), ('?b2', 'block'), ('?a', 'arm')]), 
                    'preconditions': '(and\n    (holding ?a ?b1) ; The arm is holding the top block\n    (clear ?b2) ; The bottom block is clear\n)', 
                    'effects': '(and\n    (not (holding ?a ?b1)) ; The arm is no longer holding the top block\n    (on ?b1 ?b2) ; The top block is now on the bottom block\n    (not (clear ?b2)) ; The bottom block is no longer clear\n)'}

Action Parameters
-------------------------------------------------------
**Action Parameters** are formatted as `OrderedDict` type.

For example, **formalize_parameters()** takes the LLM output: ::

    ### Action Parameters
    ```
    - ?top - block: The block being stacked on top
    - ?bottom - block: The block being stacked upon
    - ?a - arm: The arm performing the stacking action
    ```

And converts it into this: ::

    params: ParameterList = OrderedDict([('?a', 'arm'), ('?top', 'block'), ('?bottom', 'block')])

Action Preconditions
-------------------------------------------------------
**Action Preconditions** are formatted as Python string type.

For example, **formalize_preconditions()** takes the LLM output: ::

    ### Action Preconditions
    ```
    (and
        (holding ?arm ?top) ; The arm is holding the top block
        (clear ?bottom) ; The bottom block is clear
    )
    ```

And converts it into this: ::

    preconditions: str = '(and\n    (holding ?arm ?top) ; The arm is holding the top block\n    (clear ?bottom) ; The bottom block is clear\n)'

**format_preconditions**

Action Effects
-------------------------------------------------------
**Action Effects** are formatted as Python string type.

For example, **formalize_effects()** takes the LLM output: ::

    ### Action Effects
    ```
    (and
        (not (holding ?arm ?top)) ; The arm is no longer holding the top block
        (on ?top ?bottom) ; The top block is now on the bottom block
        (not (clear ?bottom)) ; The bottom block is no longer clear
    )
    ```

And converts it into this: ::

    effects: str = '(and\n    (not (holding ?arm ?top)) ; The arm is no longer holding the top block\n    (on ?top ?bottom) ; The top block is now on the bottom block\n    (not (clear ?bottom)) ; The bottom block is no longer clear\n)'}



Task Objects
-------------------------------------------------------
**Objects** are formatted as Python `dict[str,str] # {'name': 'description'}`

For example, **formalize_objects()** takes the LLM output: ::

    ## OBJECTS
    ```
    blue_block - object
    red_block - object
    yellow_block - object
    green_block - object
    ```

And converts it into this: ::

    objects: dict[str,str] = {'blue_block': 'object', 'red_block': 'object', 'yellow_block': 'object', 'green_block': 'object'}

Task Initial States
-------------------------------------------------------
**Initial States** are formatted as Python `list[dict[str,str]] # essentially [{predicate,params,neg}]`

For example, **formalize_initial_state()** takes the LLM output: ::

    ## INITIAL
    ```
    (on_top blue_block red_block): blue_block is on top of red_block
    (on_top red_block yellow_block): red_block is on top of yellow_block
    (on_table yellow_block): yellow_block is on the table
    (on_table green_block): green_block is on the table
    (clear yellow_block): yellow_block is clear
    (clear green_block): green_block is clear
    (not clear red_block): red_block is not clear
    ```

And converts it into this: ::

    initial: list[dict[str,str]] = [
            {'name': 'on_top', 'params': ['blue_block', 'red_block'], 'neg': False}, 
            {'name': 'on_top', 'params': ['red_block', 'yellow_block'], 'neg': False}, 
            {'name': 'on_table', 'params': ['yellow_block'], 'neg': False}, 
            {'name': 'on_table', 'params': ['green_block'], 'neg': False}, 
            {'name': 'clear', 'params': ['yellow_block'], 'neg': False}, 
            {'name': 'clear', 'params': ['green_block'], 'neg': False}, 
            {'name': 'clear', 'params': ['red_block'], 'neg': True}
        ]


Task Goal States
-------------------------------------------------------
**Goal States** are formatted as Python `list[dict[str,str]] # essentially [{predicate,params,neg}]`

For example, **formalize_goal_state()** takes the LLM output: ::

    ## GOAL
    ```
    (AND ; all the following should be done
        (on_top red_block green_block) ; red block is on top of green block
    )
    ```

And converts it into this: ::

    goal: list[dict[str,str]] = [{'name': 'on_top', 'params': ['red_block', 'green_block']}]