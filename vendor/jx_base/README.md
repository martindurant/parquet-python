
## Some help for the programmer

Some nomenclature is required to help follow the logic of these modules

### Table

Same as with database terminology; it is a single, unordered, set of rows;

### Schema

A set of columns that describe all the (possibly optional) properties available on all rows of a table.

### Facts

Represents the multiple tables in the hierarchical database

### Snowflake

JSON Query Expressions are used the query hierarchical databases. The relations in a hierarchical database are limited to a tree; the path between any two tables is unique; in a query, no matter which table is "origin", any column in the hierarchical database can be accessed using a unique combination of joins with the origin.

With this in mind, a Snowflake is a list of all columns, for all the tables, in the hierarchical database.

### Container

Datastore that has multiple facts

### Namespace

Metadata for a container: Information on multiple snowflakes.

  