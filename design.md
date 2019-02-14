# Design Goals

This is an evolving document about the named tensor library. 

## Goals

* No broadcasting or 1-dims. Broadcasting happens by set union. 


## Argument names

* All named dimension references should be called `dims` or `dim`.

* All named dimension constructors should be called `names` or `name`.

## Ordering

* The goal is to make named tensor a completely non-ordered library. The user should not have to reason about the position of dimensions. (Currently this is exposed through some functions but we are trying to remove it.)


## Strategy

* Whenever possible do not add any new functions to the torch standard lib. Exceptions at the moment are stack/split/dot.

* Methods only change by replacing `dim` arguments or adding addition `name` arguments.

* Class only change by adding an additional `spec` function. 

* Methods should never expect arguments of a certain name or order. This needs to be declared explicitly by the user through `dim` or `spec`.

