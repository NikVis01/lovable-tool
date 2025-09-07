import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def drop_gds_graph(driver, name: str):
    with driver.session() as session:
        try:
            result = session.run("CALL gds.graph.exists($name)", { 'name': name })
            if result.single()["exists"]:
                session.run("CALL gds.graph.drop($name)", { 'name': name })
        except Exception:
            pass


def project_gds_graph_for_leiden(driver, name: str = 'candidates_with_relationships'):
    with driver.session() as session:
        session.run(
            """
            CALL gds.graph.project(
                $name,
                { Candidate: { properties: ['years_experience'] } },
                { SIMILAR_TO: { properties: ['similarity'], orientation: 'UNDIRECTED' } }
            )
            """,
            { 'name': name }
        )


def assign_cosine_components(driver, df, similarity_matrix, id_col: str, threshold: float):
    n = len(df)
    if similarity_matrix is None:
        raise ValueError("similarity_matrix is required")

    communities = [-1] * n

    def dfs(start: int, cid: int):
        stack = [start]
        communities[start] = cid
        while stack:
            u = stack.pop()
            for v in range(n):
                if v == u or communities[v] != -1:
                    continue
                if similarity_matrix[u, v] > threshold:
                    communities[v] = cid
                    stack.append(v)

    cid = 0
    for i in range(n):
        if communities[i] == -1:
            dfs(i, cid)
            cid += 1

    with driver.session() as session:
        for idx, comm in enumerate(communities):
            node_id = int(df.iloc[idx][id_col])
            session.run(
                "MATCH (c:Candidate {id: $id}) SET c.community = $community",
                { 'id': node_id, 'community': int(comm) }
            )

    return communities


def assign_mutual_knn_components(driver, df, similarity_matrix, id_col: str, k: int):
    if similarity_matrix is None:
        raise ValueError("similarity_matrix is required")
    n = len(df)
    k = max(1, min(k, max(1, n - 1)))

    topk = []
    for i in range(n):
        sims = similarity_matrix[i].copy()
        sims[i] = -1.0
        idxs = np.argsort(-sims)[:k]
        topk.append(set(int(j) for j in idxs))

    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in topk[i]:
            if i in topk[j]:
                adj[i].append(j)
                adj[j].append(i)

    communities = [-1] * n
    cid = 0
    for i in range(n):
        if communities[i] != -1:
            continue
        stack = [i]
        communities[i] = cid
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if communities[v] == -1:
                    communities[v] = cid
                    stack.append(v)
        cid += 1

    with driver.session() as session:
        for idx, comm in enumerate(communities):
            node_id = int(df.iloc[idx][id_col])
            session.run(
                "MATCH (c:Candidate {id: $id}) SET c.community = $comm",
                { 'id': node_id, 'comm': int(comm) }
            )

    return communities


def label_nodes_by_community(driver, df, communities, id_col: str, max_old_labels: int = 100):
    with driver.session() as session:
        for k in range(max_old_labels):
            lbl = f"Community{k}"
            try:
                session.run(f"MATCH (c:Candidate:{lbl}) REMOVE c:{lbl}")
            except Exception:
                pass
        for idx, comm in enumerate(communities):
            lbl = f"Community{int(comm)}"
            node_id = int(df.iloc[idx][id_col])
            session.run(
                f"MATCH (c:Candidate {{id: $id}}) SET c:{lbl}",
                { 'id': node_id }
            )


def leiden_write_final(driver, df, id_col: str, random_seed: int = 23, graph_name: str = 'candidates_with_relationships'):
    drop_gds_graph(driver, graph_name)
    project_gds_graph_for_leiden(driver, graph_name)
    with driver.session() as session:
        result = session.run(
            "CALL gds.leiden.stream($name, $cfg) "
            "YIELD nodeId, communityId RETURN gds.util.asNode(nodeId).id AS candidate_id, communityId",
            { 'name': graph_name, 'cfg': { 'relationshipWeightProperty': 'similarity', 'randomSeed': random_seed } }
        )
        comm_map = { int(r['candidate_id']): int(r['communityId']) for r in result }
        for cand_id, community_id in comm_map.items():
            session.run(
                "MATCH (c:Candidate {id: $id}) SET c.community = $community",
                { 'id': cand_id, 'community': community_id }
            )

    communities = []
    with driver.session() as session:
        for idx in range(len(df)):
            node_id = int(df.iloc[idx][id_col])
            rec = session.run("MATCH (c:Candidate {id: $id}) RETURN c.community AS community", { 'id': node_id }).single()
            communities.append(int(rec['community']) if rec and rec['community'] is not None else 0)
    return communities


def leiden_write_intermediate_target(driver, df, id_col: str, target: int = 4, random_seed: int = 23, graph_name: str = 'candidates_with_relationships'):
    drop_gds_graph(driver, graph_name)
    project_gds_graph_for_leiden(driver, graph_name)
    with driver.session() as session:
        result = session.run(
            "CALL gds.leiden.stream($name, $cfg) "
            "YIELD nodeId, communityId, intermediateCommunityIds "
            "RETURN gds.util.asNode(nodeId).id AS candidate_id, communityId, intermediateCommunityIds",
            { 'name': graph_name, 'cfg': { 'relationshipWeightProperty': 'similarity', 'randomSeed': random_seed, 'includeIntermediateCommunities': True } }
        )
        rows = [r.data() for r in result]

    level_to_set = {}
    for r in rows:
        levels = r['intermediateCommunityIds'] or []
        for lvl, cid in enumerate(levels):
            level_to_set.setdefault(lvl, set()).add(int(cid))
    if not level_to_set:
        return leiden_write_final(driver, df, id_col, random_seed, graph_name)

    best_level, best_count = None, None
    for lvl, s in level_to_set.items():
        cnt = len(s)
        if best_level is None or abs(cnt - target) < abs(best_count - target):
            best_level, best_count = lvl, cnt

    with driver.session() as session:
        for r in rows:
            cid = int(r['candidate_id'])
            levels = r['intermediateCommunityIds'] or []
            chosen = int(levels[best_level]) if best_level < len(levels) else int(r['communityId'])
            session.run(
                "MATCH (c:Candidate {id: $id}) SET c.community = $community",
                { 'id': cid, 'community': chosen }
            )

    communities = []
    with driver.session() as session:
        for idx in range(len(df)):
            node_id = int(df.iloc[idx][id_col])
            rec = session.run("MATCH (c:Candidate {id: $id}) RETURN c.community AS community", { 'id': node_id }).single()
            communities.append(int(rec['community']) if rec and rec['community'] is not None else 0)
    return communities


