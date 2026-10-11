// =============================================================================
// Neo4j Fabric Composite Database Initialization
// Maps coordinator database queries across the 4 physical shard backends
// =============================================================================

CREATE COMPOSITE DATABASE fabric IF NOT EXISTS;

// Register physical shard aliases
CREATE ALIAS fabric.shard0 FOR DATABASE shard0 AT 'bolt://neo4j-shard-0:7687'
USER 'neo4j' PASSWORD 'password123';

CREATE ALIAS fabric.shard1 FOR DATABASE shard1 AT 'bolt://neo4j-shard-1:7687'
USER 'neo4j' PASSWORD 'password123';

CREATE ALIAS fabric.shard2 FOR DATABASE shard2 AT 'bolt://neo4j-shard-2:7687'
USER 'neo4j' PASSWORD 'password123';

CREATE ALIAS fabric.shard3 FOR DATABASE shard3 AT 'bolt://neo4j-shard-3:7687'
USER 'neo4j' PASSWORD 'password123';

