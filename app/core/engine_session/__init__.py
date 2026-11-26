from __future__ import annotations

from .plain_form import (PlainProject, _plain_teams, digest, fingerprint, plain_projects,
                         projects_from_ids, projects_to_ids, team_name_at, team_names_for,
                         unmatched_ids)
from .staleness import _roster_matcher, match_team_reports, report_matches_teams, stale_report_ids
from .summaries import (has_earlier_projects, section_reunions, show_section_reunions,
                        summarize_reports, tagged_violations, violations_from_reports)
from .scored_rules import (RULES_FROM_DEFAULTS, RULES_FROM_EARLIER, RULES_FROM_RUN,
                           RULES_FROM_SESSION, _policy_dict, _reports_signature, as_policy,
                           run_holds_reports, same_scoring, scoring_signature)
from .repeat_counting import (allocation_pair_history, cap_label, history_ids,
                              history_position, history_rules_text, other_projects_history,
                              repeat_cap, shared_team_counts)

from ...engine.runner import run_allocation
from ...engine.sizing import project_config
from .state import META_KEY, REPORTS_KEY, ROSTER_KEY, SCORED_POLICY_KEY, _session
from .roster_cache import (_data_signature, active_roster, attribute_kinds, current_policy,
                           distinct_value_counts, invalidate_roster_cache, roster_frame,
                           save_policy)
from .engine_runs import (capped_policy, evaluate_projects, imported_prior_projects,
                          project_specs, run_engine)
from .scoring_session import (_rules_info, active_rules, clear_active_reports,
                              inferred_scored_policy, rules_of_active_reports,
                              rules_source_text, scored_policy, set_active_reports)
from .repeat_explanations import (REPEAT_EXPLANATIONS_KEY, _search_stats, explain_repeat_pairs,
                                  search_base_run)
from .allocation_session import (ActiveAllocation, activate_allocation, active_allocation,
                                 active_reports, build_formation_result, clear_active_allocation,
                                 has_active_allocation, set_result_meta, unsaved_edits)
from .roster_change import (OTHER_DATASET_TEXT, departed_notice, departed_students,
                            formed_on_other_dataset, refresh_imported_projects, roster_notice)

__all__ = [
    "PlainProject", "_plain_teams", "digest", "fingerprint", "plain_projects",
    "projects_from_ids", "projects_to_ids", "team_name_at", "team_names_for", "unmatched_ids",
    "_roster_matcher", "match_team_reports", "report_matches_teams", "stale_report_ids",
    "has_earlier_projects", "section_reunions", "show_section_reunions", "summarize_reports",
    "tagged_violations", "violations_from_reports",
    "RULES_FROM_DEFAULTS", "RULES_FROM_EARLIER", "RULES_FROM_RUN", "RULES_FROM_SESSION",
    "_policy_dict", "_reports_signature", "as_policy", "run_holds_reports", "same_scoring",
    "scoring_signature",
    "allocation_pair_history", "cap_label", "history_ids", "history_position",
    "history_rules_text", "other_projects_history", "repeat_cap", "shared_team_counts",
    "run_allocation", "project_config",
    "META_KEY", "REPORTS_KEY", "ROSTER_KEY", "SCORED_POLICY_KEY", "_session",
    "_data_signature", "active_roster", "attribute_kinds", "current_policy",
    "distinct_value_counts", "invalidate_roster_cache", "roster_frame", "save_policy",
    "capped_policy", "evaluate_projects", "imported_prior_projects", "project_specs", "run_engine",
    "_rules_info", "active_rules", "clear_active_reports", "inferred_scored_policy",
    "rules_of_active_reports", "rules_source_text", "scored_policy", "set_active_reports",
    "REPEAT_EXPLANATIONS_KEY", "_search_stats", "explain_repeat_pairs", "search_base_run",
    "ActiveAllocation", "activate_allocation", "active_allocation", "active_reports",
    "build_formation_result", "clear_active_allocation", "has_active_allocation",
    "set_result_meta", "unsaved_edits",
    "OTHER_DATASET_TEXT", "departed_notice", "departed_students", "formed_on_other_dataset",
    "refresh_imported_projects", "roster_notice",
]
