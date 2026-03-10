from red_team_tests.prompt_injection_tests import run_red_team_tests

def compute_red_team_metrics():
    results = run_red_team_tests()

    total = len(results)
    blocked = sum(1 for r in results if r["blocked"])
    failures = total - blocked
    success_rate = blocked / total if total else 0

    return {
        "total_prompts": total,
        "blocked": blocked,
        "failures": failures,
        "success_rate": success_rate
    }
