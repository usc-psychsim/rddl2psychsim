> RDDL2PsychSim is a python framework for converting RDDL domains into PsychSim    

[TOC]

# Introduction

## Installation

```bash
pip install git+https://github.com/usc-psychsim/rddl2psychsim.git
```

This will install the `rddl2psychsim` package in the current python environment as well as all required packages.

## Requirements

- PsychSim: https://github.com/usc-psychsim/
- pyrddl: https://github.com/usc-psychsim/pyrddl.git (forked from original repository at https://github.com/thiagopbueno/pyrddl to support more language elements)
- NumPy: https://numpy.org/

## Usage

TODO

# Supported Language Elements

## Fluents (variables)

- `non-fluent`, converted into constants to be used for the definition of dynamics for other variables 

  - <u>Note:</u> no PsychSim features are created for non-fluents, *i.e.*, they are used only *during* conversion to PsychSim 

- ` state-fluent`, `interm-fluent`, and `observ-fluent`, all converted into PsychSim features
  
  - <u>Note:</u> `observ-fluent` variables are automatically added to agent's `omega` if `partially-observed` is specified in the `requirements` section
  
- `action-fluent`, converted into PsychSim action

  <u>Notes:</u>

  - Supports parameterized (n-arity) fluents (predicates or functions) by creating state features in PsychSim for *all* combinations of the parameters in the provided instance

  - Parameters can be of a constant type (`enum` or `object`, see below) or fluents (variable or constant) whose type has a finite domain such that they can be enumerated at conversion time

  - Fluent stratification level from RDDL file is used to define the following hierarchy among features:

    1. Actions
    2. State fluents
    3. Intermediate fluents (ordered according to the associated stratification `level` parameter)
    4. Observable fluents

    When creating the dynamics' tree for a feature in PsychSim, the future value (current update step) of other features of a *strictly lower* level in the hierarchy is considered, otherwise their past value (previous updated step) is used

## Fluent Types

- boolean: `true` and `false` constants
- integers
- reals, converted to python floats
- enumerated types via the `enum` definition in the `types` section. These correspond to domain-level constant types
  - <u>Note:</u> enums have no boolean or arithmetic evaluation, but can be used as constants in variable attribution and in relational expressions
- object types, via the `object` declaration in the `types` section and further definition in the `non-fluents` section. These correspond to instance-level constant types. 
  - <u>Note:</u> for conversion to PsychSim purposes, objects are treated just like an enumerated type

## Logical Expressions

- And (`^`)

- Or (`|`)

- Not (`∼`)

- Implies (`=>`)

- Equivalence (`<=>`)

- Quantification over variables using:

  - `forall` , corresponding to a conjunction of the provided expression iterated over the parameters

  - `exists`, corresponding to a disjunction of the provided expression iterated over the parameters

    <u>Notes:</u>

    - sub-expression can be of arithmetic, boolean or relational type, but *not* a control statement (`if`, ` switch`)
    - depending on the sub-expression type, a different underlying PWL tree representation will be created. Namely, numerical and boolean expressions (linear functions) will be used to build a single PWL tree, while relational expressions will be converted in nested `if` statements

## Arithmetic Expressions

- Sum (`+`) and Subtraction(`-`) can be used arbitrarily between numerical constants and variables (fluents)

- Multiplication (` ∗`) can only be used between one or two numerical constants or between a numerical constant and a variable (fluent)

- Division (` /`) can only be used between two numerical constants or between a variable (fluent) on the left-hand side and a constant on the right-hand side of an expression

- Aggregation over object types using:

  - `sum`, supports only numerical expressions (linear functions) as sub-expression, e.g.:

    ```yacas
    p' = sum_{?x : obj, ?y : obj}[ q(?x) + 2 * r(?y) ];
    ```

  - `prod`, supports only linear functions using numerical constants and constant variables (non-fluents) as sub-expression, e.g.:

    ```yacas
    p' = prod_{?x : obj}[ 2 * CONST(?x) ];
    ```

## Relational expressions

- Equality (`==`) and inequality (`∼=`) between any numerical variable or expressions 
- Comparison (`<`, `>`, `<=`, `>=`) between any numerical variable or expressions 
- <u>Note:</u> these expressions are always converted to a PWL tree representation. If used for variable attribution, their truth value is returned in the corresponding branch. However, they cannot be combined with arithmetic expressions

## Conditional/Control expressions

- `if`-`then`-`else`, including nested if statements

- `switch`-`case`-`default`: in general builds a non-binary PWL decision tree with branches for the different `case` values. 

  <u>Notes:</u> 

  - the `switch` expression, as well as the `case` conditions can be any arithmetic expression, however depending on the expression type a different underlying PWL tree representation will be created. Namely, constant conditional values will be used to build a single PWL tree, while expressions involving variables (fluents) will be converted in (possibly nested) `if` statements
  - `default` branch specification is *always* required

## Probability Distributions

Creates a deterministic/stochastic effect for the following distributions:

- `KronDelta(v)`: places all probability mass on its discrete argument `v`, discrete sample is thus deterministic

- `DiracDelta(v)`: places all probability mass on its continuous argument `v`, continuous sample is thus deterministic

- `Bernoulli(p)`: samples a boolean with probability of true given by parameter $p\in[0,1]$

- `Discrete(var-name,p)`: samples an enumerated value with probability vector $p$ , with $(\sum_i p_i=1)$

  - <u>Note:</u> only constant probabilities are allowed. 
    But, for example the following expression:

    ```yacas
    Discrete(enum_level,
      @low : if (p >= 2) then 0.5 else 0.2,
      @medium : if (p >= 2) then 0.2 else 0.5,
      @high : 0.3);
    ```

    could be transformed into:

    ```yacas
    if (q >= 2) then 
    	Discrete(enum_level, @low : 0.5, @medium : 0.2, @high : 0.3)
    else 
    	Discrete(enum_level, @low : 0.2, @medium : 0.5, @high : 0.3);
    ```

to achieve the same effect

- `Normal(m,s)`: samples a continuous value from a *discrete approximation* of a Normal distribution $\mathcal{N}(\mu,\sigma^{2})$ with mean $\mu=$`m` and standard deviation $\sigma=$`s`. 

  <u>Notes:</u>

  - supports arbitrary arithmetic expressions for `m` and `s`, as long as they define linear functions
  - in RDDL's definition, `s` defines the variance, not the standard deviation. But, to allow for using numerical expressions, it needs to be the standard deviation
  - the number of bins/values for the discrete approximation for Normal distributions in a domain can be set via `Converter` constructor parameter `normal_bin`. Another parameter $\tau$, defined via  `normal_stds`, stipulates the finite range of values sampled from the distribution, namely $[\mu-(\tau\sigma),\mu+(\tau\sigma)]$. See example at: `examples/normal_distribution.py`

- `Poisson(l)`: samples a value from a *discrete approximation* of a Poisson distribution with rate parameter $\lambda=$`l` per fixed time interval. 

  <u>Notes:</u>

  - supports arbitrary arithmetic expressions for `l`, as long as they define linear functions
  - the distribution is approximated via a Normal distribution $\mathcal{N}(\lambda,\sqrt{\hat{\lambda}})$, where $\hat{\lambda}$ is the expected rate of Poisson distributions for this domain. This parameter can be defined via the  `poisson_exp_rate` constructor of the ` Converter` constructor
  - due to this approximation, sampling from the distribution might return a value in $\mathbb{R}$ rather than $\mathbb{N}$, so further adjustments might be required

## State and Action Constraints

The converter supports state and action constraints as defined in the `state-action-constraints` and `action-preconditions` sections. However, *most* are treated as *assertions* rather than something that the converter uses to actively constrain PsychSim dynamics, features' values, etc. 

Constraints involving constants / non-fluents are verified at conversion time, while other constraints, possibly involving actions, are verified by calling the `verify_constraints()` method of the converter object.

If the converter constructor is invoked with `const_as_assert=True` (default), then an `AssertionError` is thrown whenever a constraint is unsatisfied, otherwise a message is sent via `logging.info`.

### Action Legality

One special type of constraints in the `state-action-constraints` or `action-preconditions` sections allows defining legality conditions for actions:

- For non-parameterized actions, this can be achieved through *implication* expressions in the form: `action => legality_expression`. For example, if `act => p <= 1;` is provided in the constraints section, then action `act` will be legal *iff* feature `p` is less than or equal to 1
- Similarly, for parameterized actions, one can define constraints in the form `forall_{?o: obj}[ action(?o) => legality_expresion ];` 

## Action-Conditioned Dynamics

Usually, dynamics to update a fluent defined inside the `cpfs` section of a RDDL domain are assigned to the "world" in PsychSim after conversion, meaning that the corresponding feature dynamics are evaluated at each time step. However, for better efficiency, one can set dynamics conditioned on the execution of some action. This can reduce the number of actions available to the agent at each step.

This is achieved by having `if` statements in the dynamics expressions of fluents where the action is the only element in the condition expression, i.e., expressions in the form: `if (act) then act_dyn_expr else world_dyn_exp`. This means that `act_dyn_expr` is only going to be evaluated when action `act` is executed, otherwise `world_dyn_exp` will be used to update the feature's value. We can define multiple action-dependent dynamics for the same feature by using nested if's, e.g.:

```yacas
x' = if (go_right) then 
        x + 1
    else if (go_left) then 
        x - 1
    else 
        x;
```

## Multiagent Domains

Multiagent scenarios can be created by defining a special type of object in the RDDL `types` section with the name `agent`. PsychSim agents are then created for each object type defined in the `non-fluents` section. For example, 

```yacas
domain my_domain {
    types { agent : object; ...};
    ...
}
...
non-fluents my_nf { 
    domain = my_domain;
    objects { agent: Agent1, Agent2; }; 
}
```

will create a world populated by two agents named `Agent1` and `Agent2`. 

If the `agent` object type is not specified, then a single PsychSim agent will be created by default. All actions defined will be associated with that agent and all features (fluents) will be associated with the world.

### Agent State

By defining fluents parameterized on the `agent` object type in a multiagent domain we can create PsychSim agents with state. For example, we could create a fluent ` pos(agent) : { state-fluent, int, default = 0 };` in the `pvariables` section of the domain and then initialize each agent's position to a different value in the `instance` definition. 

### Agent Actions

Similarly, we can parameterize each action with an agent, e.g., ` act(agent) : { action-fluent, bool};`.

<u>Note:</u> in terms of the underlying PsychSim conversion, this is equivalent to *not* specifying the `agent` type parameter, i.e., all actions will be created for all the agents. However, the advantage of the action parameterization is that we can specify in the RDDL definition both *action constraints* (legality), via

```yacas
forall_{?a : agent}}[ act(?a) => ... ];
```

and *action-conditioned dynamics*, via

````yacas
x'(?a) = if ( action1(?a) ) then ... else if ( action2(?a) ) then ... else ...;
````

for agent-parameterized fluents, or via

```yacas
x' = if ( exists_{?a: agent}[action1(?a)] ) then 
    ... 
else if ( exists_{?a: agent}[action2(?a)] ) then 
    ... 
else 
    ...;
```

for non-agent-parameterized (world) fluents.

We can also parameterize fluents and actions with other object types. The order in which the `agent` parameter appears is not relevant, i.e., `pos(agent, x_coord, y_coord)` is equivalent to `pos(x_coord, agent, y_coord)`.

<u>Note:</u> however the order of the other parameters *does* matter.

### Concurrency

RDDL concurrency is translated into PsychSim in terms of "agent turns", i.e., who acts in parallel and what is the order among agents at each decision step. By default, a world is created with a *sequential* turn order among the agents, where the agents' turn order is defined according to the order in which the `agent` object types are specified in the `objects` section of the `non-fluents` instance. In the example above, `Agent1` would act first, then `Agent2`, then `Agent1` again and so on.

However, if `concurrent` is defined in the domain's `requirements` section, then the agents can act in parallel (concurrently). If nothing else is specified, then *all* agents have the same turn order. If `max-nondef-actions=<INT>` is specified in the `instance` definition, then that number of agents will act in parallel (same turn), and again the order will be as specified by the `agent` object's definition. For example, 

```yacas
non-fluents my_nf { objects { agent: Agent1, Agent2, Agent3; }; ... } 
instance my_inst { max-nondef-actions = 2; ... }
```

would result in `Agent1` and `Agent2` acting in parallel in the first turn, then `Agent3` acting in the second turn, and so on.

# References

- RDDL manual: http://users.cecs.anu.edu.au/~ssanner/IPPC_2011/RDDL.pdf
- RDDL Tutorial: https://sites.google.com/site/rddltutorial/rddl-language-discription
- PsychSim repository: https://github.com/usc-psychsim/
- PsychSim manual: https://psychsim.readthedocs.io/