#!/usr/bin/env python
# coding: utf-8

# # Explore Round 1 Data Patterns
# An exploration of the round 1 data to understand prominent patterns.

# In[1]:


import pathlib

import kaleido
import pandas as pd
import plotly.express as px
import plotly.io as pio
import pyarrow.parquet as pq
from umap import UMAP


# ## Find the root of the git repo on the host system

# In[2]:


# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")


# # Inputs

# In[3]:


pilot_path = root_dir / "big_drive/alsf/ALSF_pilot_data"

round1_profiles_path = (
    pilot_path / "preprocessed_profiles_SN0313537/single_cell_profiles"
).resolve(strict=True)

common_cols = None

for pq_file in list(round1_profiles_path.rglob("*feature_selected.parquet")):
    cols = pq.read_schema(pq_file).names
    if common_cols is None:
        common_cols = set(cols)

    else:
        common_cols &= set(cols)

round1df = pd.concat(
    [
        pd.read_parquet(pq_file, columns=common_cols)
        for pq_file in round1_profiles_path.rglob("*feature_selected.parquet")
    ],
    axis=0,
)


# In[4]:


round1_figure_path = pathlib.Path("round1_figures")

int_figure_path = round1_figure_path / "interactive_figures"
int_figure_path.mkdir(parents=True, exist_ok=True)

static_figure_path = round1_figure_path / "static_figures"
static_figure_path.mkdir(parents=True, exist_ok=True)


# # High-Level Data Characteristics

# In[5]:


print(round1df.shape)


# In[6]:


print("Are there any NaN entries?")
print(f"Answer: {round1df.isna().any().any()}")


# ## Metadata Exploration

# In[7]:


round1df["Metadata_Plate"].unique()


# In[8]:


round1df["Metadata_time_point"].unique()


# In[9]:


print("\nUnique timepoints per plate:")
print(round1df[["Metadata_Plate", "Metadata_time_point"]].drop_duplicates())


# In[10]:


print("\nUnique Seeding Densities:")
round1df["Metadata_seeding_density"].unique()


# In[11]:


round1df["Metadata_seeding_density"].value_counts()


# In[12]:


round1df["Metadata_cell_line"].unique()


# In[13]:


round1df["Metadata_Image_Count_Cells"].nunique()


# In[14]:


round1df["Metadata_Plate"].value_counts()


# # UMAP Morphology Figures
# Sampled to show morphology relationships between experimental conditions

# In[15]:


vdf = round1df.groupby(
    [
        "Metadata_Plate",
        "Metadata_cell_line",
        "Metadata_seeding_density",
        "Metadata_time_point",
    ]
).sample(n=50, random_state=0)

umap_obj = UMAP(random_state=0)
umapdf = umap_obj.fit_transform(
    vdf.loc[:, ~vdf.columns.str.contains("metadata", case=False)]
)
umapdf = pd.DataFrame(umapdf, columns=["umap0", "umap1"])
umapdf = umapdf.assign(
    Metadata_Plate=vdf["Metadata_Plate"].reset_index(drop=True),
    Metadata_cell_line=vdf["Metadata_cell_line"].reset_index(drop=True),
    Metadata_seeding_density=vdf["Metadata_seeding_density"].reset_index(drop=True),
    Metadata_time_point=vdf["Metadata_time_point"].reset_index(drop=True),
)
umapdf["Metadata_seeding_density"] = umapdf["Metadata_seeding_density"].astype(str)
umapdf["Metadata_time_point"] = umapdf["Metadata_time_point"].astype(str)


# In[16]:


# Needed to display in chrome (if not installed)
kaleido.get_chrome_sync()


# In[17]:


red_yellow_colors = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#f03b20", "#bd0026"]

fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_Plate",
    title="UMAP by Plate",
    color_discrete_sequence=red_yellow_colors,
    marginal_x="violin",
    marginal_y="violin",
)

fig.update_layout(
    font=dict(size=22, color="black"),
    legend=dict(font=dict(size=20)),
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
)

fig.show()
fig.write_image(static_figure_path / "round1_plate_umap.png", width=2000, height=1200)
fig.write_html(
    int_figure_path / "round1_plate_umap.html",
    full_html=True,
    include_plotlyjs="embed",
)


# In[18]:


fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_cell_line",
    title="UMAP by Cell Line",
    color_discrete_sequence=px.colors.qualitative.Dark24,
)

fig.update_layout(
    font=dict(size=22, color="black"),
    legend=dict(font=dict(size=20)),
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
)

fig.show()
fig.write_image(
    static_figure_path / "round1_cell_line_umap.png", width=2000, height=1200
)
fig.write_html(
    int_figure_path / "round1_cell_line_umap.html",
    full_html=True,
    include_plotlyjs="embed",
)


# In[19]:


yellow_green_colors = ["#ffffcc", "#c2e699", "#78c679", "#31a354", "#006837"]

fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_seeding_density",
    title="UMAP by Seeding Density",
    color_discrete_sequence=yellow_green_colors,
    marginal_x="violin",
    marginal_y="violin",
)

fig.update_layout(
    font=dict(size=22, color="black"),
    legend=dict(font=dict(size=20)),
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
)

fig.show()
fig.write_image(
    static_figure_path / "round1_seeding_density_umap.png", width=2000, height=1200
)
fig.write_html(
    int_figure_path / "round1_seeding_density_umap.html",
    full_html=True,
    include_plotlyjs="embed",
)


# In[20]:


yellow_green_short_colors = ["#ffeda0", "#feb24c", "#f03b20"]

fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_time_point",
    title="UMAP by Time Point",
    color_discrete_sequence=yellow_green_short_colors,
)

fig.update_layout(
    font=dict(size=22, color="black"),
    legend=dict(font=dict(size=20)),
    xaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
    yaxis=dict(
        title_font=dict(size=20),
        tickfont=dict(size=16),
    ),
)

fig.show()
fig.write_image(
    static_figure_path / "round1_time_point_umap.png", width=2000, height=1200
)
fig.write_html(
    int_figure_path / "round1_time_point_umap.html",
    full_html=True,
    include_plotlyjs="embed",
)


# # Visualize Cell Counts per Plate and Well

# In[21]:


global_max = round1df.groupby(["Metadata_Plate", "Metadata_Well"]).size().max()

for plate in round1df["Metadata_Plate"].unique():
    plate_df = round1df[round1df["Metadata_Plate"] == plate]

    well_counts = plate_df[["Metadata_Well"]].value_counts().reset_index(name="count")

    meta_per_well = (
        plate_df.groupby("Metadata_Well")[
            ["Metadata_seeding_density", "Metadata_cell_line", "Metadata_time_point"]
        ]
        .first()
        .reset_index()
    )

    meta_per_well = meta_per_well.rename(
        columns={
            "Metadata_seeding_density": "Seeding Density",
            "Metadata_cell_line": "Cell Line",
            "Metadata_time_point": "Time Point",
        }
    )

    well_counts = well_counts.merge(meta_per_well, on="Metadata_Well").sort_values(
        "Metadata_Well"
    )

    fig = px.bar(
        well_counts,
        x="Metadata_Well",
        y="count",
        hover_data=["Seeding Density", "Cell Line", "Time Point"],
        title=f"Cell Count per Well - Plate {plate}",
        labels={
            "Metadata_Well": "Well",
        },
    )

    fig.update_layout(
        font=dict(size=18, color="black"),
        xaxis_tickangle=-45,
        yaxis_range=[0, global_max * 1.05],
    )

    fig.show()
    fig.write_image(
        static_figure_path / f"{plate}_well_cell_count.png",
        width=2000,
        height=1200,
    )

    fig.write_html(
        int_figure_path / f"{plate}_well_cell_count.html",
        full_html=True,
        include_plotlyjs="embed",
    )

