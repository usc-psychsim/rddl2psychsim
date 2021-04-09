# Supported Language Elements

## Fluents (variables)

- `non-fluent`, converted into constants to be used for the definition of dynamics for other variables 
- ` state-fluent`, `interm-fluent`, and `observ-fluent`, all converted into PsychSim features
  - <u>Note:</u> `observ-fluent` variables are automatically added to agent's `omega` if `partially-observed` is specified in the `requirements` section
- `action-fluent`, converted into PsychSim action

TODO: support for n-arity predicates

## Constants

- `true`, `false` 
- integers
- reals, converted to python floats
- enumerated values via the definition in `types`. Note: enums have no boolean or arithmetic evaluation, but can be used as constants in variable attribution and in relational expressions

## Logical expressions

- And (`^`)
- Or (`|`)
- Not (`∼`)
- Implies (`=>`)
- Equivalence (`<=>`)
- TODO: quantification over variables using `forall` and `exists`

## Arithmetic expressions

- Sum (`+`) and Subtraction(`-`) can be used arbitrarily between numerical constants and variables (fluents)
- Multiplication (` ∗`) can be used between one or two numerical constants or between a numerical constant and a variable (fluent)
- Division (` /`) can only be used between two numerical constants or between a variable (fluent) on the left-hand side and a constant on the right-hand side of an expression
- TODO: aggregation over object types using `sum` and  `prod`

## Relational expressions

- Equality (`==`) and inequality (`∼=`) between any numerical variable or expressions 
- Comparison (`<`, `>`, `<=`, `>=`) between any numerical variable or expressions 
- Note: these expressions are always converted to a PWL tree representation. If used for variable attribution, their truth value is returned in the corresponding branch. However, cannot be combined with arithmetic expressions

## Conditional/Control expressions

- `if`-`then`-`else`, including nested if statements
- `switch`-`case`-`default`: in general builds a non-binary PWL decision tree with branches for the different `case` values. Note: the `switch` expression, as well as the `case` conditions can be any arithmetic expression, however depending on the expression type a different underlying PWL tree representation will be created. Namely, constant conditional values will be used to build a single PWL tree, while expressions involving variables (fluents) will be converted in (possibly nested) if statements. `default` branch specification is always required

## Probability distributions

Creates a deterministic/stochastic effect for the following distributions:

- `KronDelta(v)`: places all probability mass on its discrete argument `v`, discrete sample is thus deterministic

- `DiracDelta(v)`: places all probability mass on its continuous argument `v`, continuous sample is thus deterministic

- `Bernoulli(p)`: samples a boolean with probability of true given by parameter $p\in[0,1]$

- `Discrete(var-name,p)`: samples an enumerated value with probability vector $p$ , with $(\sum_i p_i=1)$

  - <u>Note:</u> only constant probabilities are allowed. 
    But, for example the following expression:

    ```python
    Discrete(enum_level,
      @low : if (p >= 2) then 0.5 else 0.2,
      @medium : if (p >= 2) then 0.2 else 0.5,
      @high : 0.3);
    ```

    could be transformed into:

    ```python
    if (q >= 2) then
      Discrete(enum_level, @low : 0.5, @medium : 0.2, @high : 0.3)
    else
      Discrete(enum_level, @low : 0.2, @medium : 0.5, @high : 0.3);
    ```

    to achieve the same effect

- TODO: `Normal(m,s)`: samples a continuous value from a Normal distribution with mean $\mu=$`m` and variance $\sigma^2=$`s`

- TODO: `Poisson(l)`: samples an integer value from a Poisson distribution with rate parameter $\lambda=$`l` per fixed time interval

