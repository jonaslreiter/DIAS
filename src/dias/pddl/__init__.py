"""PDDL source manipulation: text in, text out.

Import the submodules directly. `text` is pure string handling and imports
nothing, so reaching for a regex helper does not pull in numpy (`transpiler`)
or pymimir (`states`).
"""
