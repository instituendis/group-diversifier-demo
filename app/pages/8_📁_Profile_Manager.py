import copy
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import streamlit as st
import yaml

APP_DIR = Path(__file__).parent.parent
ROOT_DIR = APP_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app import demo, extensions, hosting
from app.core.config_loader import ConfigurationLoader
from app.core.data_session import (
    ensure_session_state,
    flash,
    get_config_dir,
    is_new_upload,
    mark_upload_processed,
    render_flashes,
    sync_profile_dimensions,
)
from app.core.dimension_config import DimensionConfig, Level, Profile

st.set_page_config(
    page_title="Profile Manager | Group Diversifier",
    page_icon="📁",
    layout="wide",
    initial_sidebar_state="expanded",
)

config = ensure_session_state()

if demo.classic_hidden():
    st.title("Profile Configuration Manager")
    st.info("📁 " + hosting.PROFILES_READ_ONLY)
    st.stop()
if hosting.hosted():
    st.info("📁 " + hosting.PROFILES_READ_ONLY)

loader = ConfigurationLoader(get_config_dir())

_MODE_KEY = "pm_mode"
_PROFILE_KEY = "pm_profile"
_SELECT_NEXT_KEY = "_pm_select_next"
_LEVEL_RE = re.compile(
    r"^(?:level\s*\d+\s*:\s*)?(?P<name>.+?)\s*\(\s*(?:weight\s*:\s*)?(?P<weight>-?\d+(?:\.\d+)?)\s*\)\s*$",
    re.IGNORECASE,
)


def set_mode(mode: str) -> None:
    st.session_state[_MODE_KEY] = mode


PROTECTED_PROFILE = "default"


class ProfileRefused(Exception): ...


def file_stem(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("-", "_")).lower()


def parse_levels(text: str) -> Tuple[List[Level], List[str]]:
    levels, errors = [], []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        match = _LEVEL_RE.match(line.strip())
        if not match:
            errors.append(f"Line {line_no}: expected 'Name (weight: number)', got {line.strip()!r}")
            continue
        levels.append(Level(name=match.group("name").strip(), weight=float(match.group("weight"))))
    return levels, errors


def lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def edit_key(profile_stem: str, dim_name: str, field: str) -> str:
    return f"pm_edit_{profile_stem}_{dim_name}_{field}"


def drop_edit_state(profile_stem: str, dim_name: str) -> None:
    prefix = f"pm_edit_{profile_stem}_{dim_name}_"
    for key in [k for k in list(st.session_state) if isinstance(k, str) and k.startswith(prefix)]:
        del st.session_state[key]


st.title("Profile Configuration Manager")
st.markdown(
    "Manage configuration profiles for team formation. Profiles define which "
    "dimensions and values are considered during team formation."
)
render_flashes()
st.session_state.setdefault(_MODE_KEY, "view")

st.sidebar.header("Quick Actions")
profiles = loader.load_all_profiles()
profile: Optional[Profile] = None
current_profile: Optional[str] = None
if profiles:
    queued = st.session_state.pop(_SELECT_NEXT_KEY, None)
    if queued in profiles:
        st.session_state[_PROFILE_KEY] = queued
    if st.session_state.get(_PROFILE_KEY) not in profiles:
        st.session_state[_PROFILE_KEY] = (
            config.profile_name if config.profile_name in profiles else profiles[0]
        )
    current_profile = st.sidebar.selectbox(
        "Select Profile",
        options=profiles,
        key=_PROFILE_KEY,
        on_change=set_mode,
        args=("view",),
        help="Choose a configuration profile to view or edit",
    )
    try:
        profile = loader.load_profile(current_profile)
        st.sidebar.success(f"✅ Loaded: {profile.name}")
        st.sidebar.markdown(f"**Version**: {profile.version}")
        st.sidebar.markdown(f"**Description**: {profile.description}")
        enabled_count = sum(1 for dim in profile.dimensions if dim.enabled)
        st.sidebar.metric("Enabled Dimensions", f"{enabled_count}/{len(profile.dimensions)}")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading profile: {e}")
else:
    st.sidebar.warning("⚠️ No profiles found")

st.sidebar.divider()
st.sidebar.subheader("Actions")
col1, col2 = st.sidebar.columns(2)
with col1:
    st.button("➕ Create New Profile", width="stretch", on_click=set_mode, args=("create",))
    st.button("📤 Upload Profile", width="stretch", on_click=set_mode, args=("upload",))
with col2:
    st.button("🔄 Refresh Profiles", width="stretch", on_click=set_mode, args=("view",))
st.sidebar.divider()

mode = st.session_state[_MODE_KEY]

if mode == "create":
    st.header("➕ Create New Profile")
    st.markdown("Create a new configuration profile; add or tune its dimensions afterwards.")
    new_profile_name = st.text_input(
        "Profile Name",
        placeholder="e.g., Hackathon Fun, Corporate Training",
        help="Unique name for this profile"
    )
    new_profile_description = st.text_area(
        "Description",
        placeholder="Describe the purpose of this profile...",
        help="Human-readable description of this profile"
    )
    new_profile_version = st.text_input("Version", value="1.0", help="Profile version (e.g., 1.0, 1.2.3)")
    copy_dims = st.checkbox(
        f"Start with the dimensions of '{current_profile}'" if current_profile else "Start empty",
        value=profile is not None,
        disabled=profile is None,
    )
    ccol1, ccol2 = st.columns(2)
    with ccol1:
        create_clicked = st.button("💾 Create Profile", type="primary")
    with ccol2:
        st.button("Cancel", on_click=set_mode, args=("view",))
    if create_clicked:
        stem = file_stem(new_profile_name.replace(" ", "_"))
        if not new_profile_name.strip() or not stem:
            st.error("❌ Profile name is required (letters, digits, - or _)")
        elif stem in profiles:
            st.error(f"❌ A profile file named '{stem}' already exists")
        else:
            new_profile = Profile(
                name=new_profile_name.strip(),
                description=new_profile_description,
                version=new_profile_version.strip() or "1.0",
                dimensions=copy.deepcopy(profile.dimensions) if (copy_dims and profile) else [],
            )
            try:
                profile_path = loader.save_profile(new_profile, stem)
            except Exception as e:
                st.error(f"❌ Error creating profile: {e}")
            else:
                flash(f"✅ Profile '{new_profile.name}' created ({profile_path.name}).")
                st.session_state[_SELECT_NEXT_KEY] = profile_path.stem
                set_mode("view")
                st.rerun()

elif mode == "upload":
    st.header("📤 Upload Custom Profile")
    st.markdown("Upload a YAML configuration file to add it to the available profiles.")
    uploaded_file = st.file_uploader(
        "Upload YAML Profile",
        type=["yaml", "yml"],
        help="Upload a YAML file containing a profile configuration"
    )
    overwrite = st.checkbox(
        "Replace an existing profile with the same file name", key="pm_upload_overwrite",
        help="Off: a profile whose name maps to an existing file is refused. The shipped "
             "default profile is never replaced.")
    st.button("Cancel", on_click=set_mode, args=("view",))
    if uploaded_file is not None and is_new_upload("profile_yaml", uploaded_file):
        try:
            content = yaml.safe_load(uploaded_file.getvalue())
            if not isinstance(content, dict):
                st.error("❌ Invalid profile: the file must contain a YAML mapping")
            elif "name" not in content:
                st.error("❌ Invalid profile: Missing 'name' field")
            elif "dimensions" not in content:
                st.error("❌ Invalid profile: Missing 'dimensions' field")
            else:
                content.setdefault("description", "")
                content.setdefault("version", "1.0")
                uploaded_profile = Profile.from_dict(content)
                stem = file_stem(str(content["name"]).replace(" ", "_")) or file_stem(Path(uploaded_file.name).stem)
                if stem == PROTECTED_PROFILE:
                    raise ProfileRefused(
                        f"'{content['name']}' maps to {PROTECTED_PROFILE}.yaml, the shipped "
                        "default profile every session starts from, which cannot be replaced. "
                        "Change the profile's 'name' in the file.")
                if stem in profiles and not overwrite:
                    raise ProfileRefused(
                        f"A profile file named '{stem}' already exists. Tick 'Replace an "
                        "existing profile…' to overwrite it, or change the profile's 'name' "
                        "in the file.")
                saved_path = loader.save_profile(uploaded_profile, stem)
                mark_upload_processed("profile_yaml", uploaded_file)
                flash(f"✅ Profile '{uploaded_profile.name}' uploaded ({saved_path.name}).")
                st.session_state[_SELECT_NEXT_KEY] = saved_path.stem
                set_mode("view")
                st.rerun()
        except ProfileRefused as e:
            st.error(f"❌ {e}")
        except yaml.YAMLError as e:
            st.error(f"❌ Invalid YAML: {e}")
        except Exception as e:
            st.error(f"❌ Error uploading profile: {e}")

elif profile is not None and current_profile:
    st.header(f"📋 Profile: {profile.name}")
    is_active = config.profile_name == current_profile
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Profile Information")
        st.markdown(f"**Description**: {profile.description}")
        st.markdown(f"**Version**: {profile.version}")
        st.metric("Total Dimensions", len(profile.dimensions))
        st.metric("Enabled Dimensions", sum(1 for d in profile.dimensions if d.enabled))
        st.caption("✅ This is the active profile of the current configuration." if is_active
                   else f"The current configuration uses profile '{config.profile_name}'.")
        if st.button("🎯 Apply to current configuration",
                     help="Replace the live configuration's dimensions (and their weights) with this profile's"):
            error = sync_profile_dimensions(config, current_profile)
            if error:
                st.error(f"❌ Could not apply profile: {error}")
            else:
                flash(f"Profile '{current_profile}' applied to the current configuration.")
                st.rerun()
    with col2:
        st.subheader("Profile Statistics")
        type_counts = {}
        for dim in profile.dimensions:
            type_counts[dim.type] = type_counts.get(dim.type, 0) + 1
        for dtype, count in type_counts.items():
            st.metric(f"{dtype.title()}", count)
        st.metric("Disabled", sum(1 for d in profile.dimensions if not d.enabled))

    st.divider()
    st.subheader("Dimensions Configuration")
    dim_data = []
    for dim in profile.dimensions:
        if dim.type == "categorical":
            options_str = ", ".join(dim.categories) if dim.categories else "(open-ended: values from data)"
        elif dim.type == "ordinal":
            options_str = ", ".join(f"{lvl.name} ({lvl.weight:g})" for lvl in dim.levels)
        elif dim.type == "multi_select":
            options_str = f"{len(dim.values)} options (max {dim.max_selections})"
        elif dim.type == "numeric":
            options_str = f"{dim.min_value} - {dim.max_value}"
        else:
            options_str = "N/A"
        dim_data.append({
            "Name": dim.name,
            "Type": dim.type,
            "Enabled": "✓" if dim.enabled else "✗",
            "Weight": float(dim.weight),
            "Options": options_str,
        })
    st.dataframe(dim_data, width="stretch", height=400)

    st.divider()
    st.subheader("Export Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("Download profile as YAML file for sharing or backup.")
    with col2:
        st.download_button(
            label="⬇️ Download YAML",
            data=yaml.dump(profile.to_dict(), default_flow_style=False, sort_keys=False, allow_unicode=True),
            file_name=f"{current_profile}.yaml",
            mime="text/yaml",
            on_click="ignore",
        )

    st.divider()
    st.subheader("Quick Edit")
    st.markdown("Select a dimension to edit its settings; **Save Changes** writes the profile file.")
    if not profile.dimensions:
        st.info("This profile has no dimensions yet. Upload a YAML file with dimensions to extend it.")
    else:
        dimension_to_edit = st.selectbox(
            "Dimension to Edit",
            options=[dim.name for dim in profile.dimensions],
            help="Choose which dimension to edit",
            key=f"pm_dimension_{current_profile}",
        )
        dim = next((d for d in profile.dimensions if d.name == dimension_to_edit), None)
        if dim:
            st.markdown(f"### Editing: {dim.name}")
            new_enabled = st.checkbox(
                "Enabled", value=dim.enabled,
                key=edit_key(current_profile, dim.name, "enabled"),
                help="Whether this dimension is used in team formation",
            )
            new_weight = st.slider(
                "Weight", min_value=0.0, max_value=200.0,
                value=float(min(max(dim.weight, 0.0), 200.0)), step=5.0,
                key=edit_key(current_profile, dim.name, "weight"),
                help="Importance weight (higher = more important)",
            )
            categories_text = levels_text = values_text = None
            max_sel = min_val = max_val = None
            if dim.type == "categorical":
                st.markdown("**Categories**")
                categories_text = st.text_area(
                    "Categories", value="\n".join(dim.categories), height=150,
                    key=edit_key(current_profile, dim.name, "categories"),
                    help="One category per line (empty = accept any value from the data)",
                )
            elif dim.type == "ordinal":
                st.markdown("**Levels**")
                levels_text = st.text_area(
                    "Levels Configuration",
                    value="\n".join(f"Level {i + 1}: {lvl.name} (weight: {lvl.weight:g})"
                                    for i, lvl in enumerate(dim.levels)),
                    height=200,
                    key=edit_key(current_profile, dim.name, "levels"),
                    help="Format: Level X: Name (weight: Y)",
                )
            elif dim.type == "multi_select":
                st.markdown("**Values**")
                values_text = st.text_area(
                    "Values", value="\n".join(dim.values), height=150,
                    key=edit_key(current_profile, dim.name, "values"),
                    help="One value per line",
                )
                max_sel = st.number_input(
                    "Max Selections", min_value=1, max_value=10,
                    value=int(min(max(dim.max_selections or 5, 1), 10)),
                    key=edit_key(current_profile, dim.name, "max_selections"),
                    help="Maximum number of values a student can select",
                )
            elif dim.type == "numeric":
                min_val = st.number_input(
                    "Minimum Value",
                    value=float(dim.min_value) if dim.min_value is not None else 0.0,
                    key=edit_key(current_profile, dim.name, "min"),
                    help="Minimum allowed value",
                )
                max_val = st.number_input(
                    "Maximum Value",
                    value=float(dim.max_value) if dim.max_value is not None else 100.0,
                    key=edit_key(current_profile, dim.name, "max"),
                    help="Maximum allowed value",
                )

            col_save, col_cancel = st.columns(2)
            with col_save:
                save_clicked = st.button("💾 Save Changes", type="primary")
            with col_cancel:
                reset_clicked = st.button("❌ Reset")
            if reset_clicked:
                drop_edit_state(current_profile, dim.name)
                flash("Changes reset to profile values", "info")
                st.rerun()
            if save_clicked:
                updated: DimensionConfig = copy.deepcopy(dim)
                updated.enabled = bool(new_enabled)
                updated.weight = float(new_weight)
                errors: List[str] = []
                if categories_text is not None:
                    updated.categories = lines(categories_text)
                if levels_text is not None:
                    updated.levels, errors = parse_levels(levels_text)
                if values_text is not None:
                    updated.values = lines(values_text)
                    updated.max_selections = int(max_sel)
                if min_val is not None and max_val is not None:
                    if min_val > max_val:
                        errors.append("Minimum value must not exceed the maximum value")
                    updated.min_value, updated.max_value = float(min_val), float(max_val)
                if not errors and not updated.is_valid():
                    errors.append(f"The {updated.type} dimension is incomplete (e.g. no levels or values)")
                if errors:
                    for error in errors:
                        st.error(f"❌ {error}")
                else:
                    edited = copy.deepcopy(profile)
                    edited.dimensions = [updated if d.name == dim.name else d for d in edited.dimensions]
                    try:
                        loader.save_profile(edited, current_profile)
                    except Exception as e:
                        st.error(f"❌ Could not save profile: {e}")
                    else:
                        drop_edit_state(current_profile, dim.name)
                        message = f"✅ Changes to {dim.name} saved to {current_profile}.yaml."
                        if is_active:
                            message += " Use 'Apply to current configuration' to use them in team formation."
                        flash(message)
                        st.rerun()

else:
    st.info("👆 Select or create a profile to get started!")
    st.divider()
    st.subheader("📚 Help")
    with st.expander("How to Create Profiles"):
        st.markdown("""
        ### Profile Structure

        A YAML profile has this structure:
        ```yaml
        name: "Profile Name"
        description: "Description of this profile"
        version: "1.0"
        dimensions:
          - name: "dimension_name"
            type: "categorical|ordinal|multi_select|numeric"
            enabled: true
            weight: 50.0
            categories:  # for categorical (empty = values taken from the data)
              - "Category1"
            levels:  # for ordinal
              - name: "Level1"
                weight: 10
            values:  # for multi_select
              - "Value1"
            max_selections: 3  # for multi_select
            min_value: 0  # for numeric
            max_value: 100  # for numeric
        ```

        ### Dimension Types

""" + extensions.text("profile.dimension_examples", """        - **Categorical**: Discrete categories (e.g., a label, a city)
        - **Ordinal**: Ordered levels (e.g., skill: Seedling < Sprout < Sapling < Oak)
        - **Multi-Select**: Multiple values with a maximum (e.g., hobbies)
        - **Numeric**: Continuous range (e.g., a score)
""") + """        """)
