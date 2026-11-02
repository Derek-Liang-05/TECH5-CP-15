"""End-to-end test of the Week 3 milestone, driven only through the web UI.

Unlike the other integration tests, the cohort is not loaded from a demo
scenario: it is typed in through the data entry forms. The test then walks
through every item of the milestone in order: data entry, run trigger,
distinct alternatives, comparison, transparent recommendations, fixed
assignments and infeasibility handling.
"""

import pathlib

import flask.testing

from app import create_app
from app.domain import models
from app.persistence import repositories


def _enter_cohort(client: flask.testing.FlaskClient) -> None:
    """Enters 1 skill, 3 projects and 5 students through the UI forms."""
    client.post("/skills/new", data={"id": "py", "name": "Python"})

    for p_id, min_cap, max_cap, py_level in [
        ("P1", 1, 2, 3),
        ("P2", 1, 2, 0),
        ("P3", 0, 2, 0),
    ]:
        client.post(
            "/projects/new",
            data={
                "id": p_id,
                "title": f"Project {p_id}",
                "min_capacity": min_cap,
                "max_capacity": max_cap,
                "req_level_py": py_level,
            },
        )

    for s_id, py_level, prefs in [
        ("S1", 4, "P1, P2, P3"),
        ("S2", 1, "P1, P3, P2"),
        ("S3", 2, "P1, P2"),
        ("S4", 5, "P2, P1, P3"),
        ("S5", 0, "P3, P2, P1"),
    ]:
        client.post(
            "/students/new",
            data={
                "id": s_id,
                "display_name": f"Student {s_id}",
                "skill_py": py_level,
                "preferences_text": prefs,
            },
        )


def _trigger_run(client: flask.testing.FlaskClient) -> models.AssignmentRun:
    """Triggers a run through the UI and returns the stored run."""
    res = client.post(
        "/runs/trigger",
        data={"primary_strategy": "PREFERENCE", "max_alternatives": 3},
        follow_redirects=True,
    )
    assert res.status_code == 200
    latest = repositories.RunRepository().list_runs(limit=1)[0]
    run = repositories.RunRepository().get_run(latest.id)
    assert run is not None
    return run


def _allocation(solution: models.Solution) -> frozenset[tuple[str, str]]:
    """Returns the set of (student, project) pairs of a solution."""
    return frozenset((a.student_id, a.project_id) for a in solution.assignments)


def test_milestone_end_to_end(tmp_path: pathlib.Path) -> None:
    """Walks through every milestone item using only the web interface."""
    app = create_app(
        {"TESTING": True, "DATABASE_PATH": tmp_path / "milestone.db"}
    )
    client = app.test_client()

    # 1. Data entry: the cohort typed into the forms is stored and shown
    _enter_cohort(client)
    assert len(repositories.StudentRepository().list_all()) == 5
    assert len(repositories.ProjectRepository().list_all()) == 3
    assert b"Student S5" in client.get("/students").data
    assert b"Project P3" in client.get("/projects").data

    # 2. Run trigger: the real engine solves the cohort
    run = _trigger_run(client)
    assert run.status == "COMPLETED"
    assert client.get(f"/runs/{run.id}").status_code == 200

    # 3. Distinct alternatives: every solution is a different allocation,
    #    and every student is assigned exactly once in each of them
    assert len(run.solutions) >= 2
    allocations = [_allocation(sol) for sol in run.solutions]
    assert len(set(allocations)) == len(allocations)
    for sol in run.solutions:
        assert sorted(a.student_id for a in sol.assignments) == [
            "S1", "S2", "S3", "S4", "S5"
        ]

    # 4. Comparison: two alternatives side by side, with deltas
    sol_a, sol_b = run.solutions[0].id, run.solutions[1].id
    comp = client.get(f"/compare?sol1={sol_a}&sol2={sol_b}")
    assert comp.status_code == 200
    assert b"Side-by-Side Solution Comparison" in comp.data
    assert b"Trade-off Findings" in comp.data

    # 5. Transparent recommendations: evidence is kept apart from advice
    rec = client.get(f"/recommendations/{run.id}")
    assert rec.status_code == 200
    assert b"Calculated Evidence" in rec.data
    assert b"Policy Advice" in rec.data

    # 6. Fixed assignments: lock S5 to its last choice; the lock must hold
    #    in every alternative of the rerun
    client.post(
        "/fixed-assignments/new",
        data={"student_id": "S5", "project_id": "P1", "status": "APPROVED"},
    )
    rerun = _trigger_run(client)
    assert rerun.id != run.id
    assert rerun.status == "COMPLETED"
    for sol in rerun.solutions:
        locked = [a for a in sol.assignments if a.student_id == "S5"]
        assert len(locked) == 1
        assert locked[0].project_id == "P1"
        assert locked[0].is_fixed

    # 7. Infeasibility: three locks on P1 exceed its maximum of 2. The run
    #    is saved as infeasible with a reason, and the page explains it
    for s_id in ("S1", "S2"):
        client.post(
            "/fixed-assignments/new",
            data={"student_id": s_id, "project_id": "P1", "status": "APPROVED"},
        )
    bad_run = _trigger_run(client)
    assert bad_run.status in ("INFEASIBLE", "INVALID_INPUT")
    assert not bad_run.solutions
    assert bad_run.diagnostics
    page = client.get(f"/runs/{bad_run.id}")
    assert page.status_code == 200
    assert bad_run.status.encode() in page.data

    # Earlier runs are kept, so the decision-maker can still go back to them
    assert repositories.RunRepository().get_run(run.id) is not None
