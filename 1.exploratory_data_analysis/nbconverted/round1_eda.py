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
    cols = pd.read_parquet(pq_file).columns
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


print("\nUnique timepoints:")
round1df["Metadata_time_point"].unique()


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


# # UMAP Figures

# In[15]:


vdf = round1df.groupby(
    ["Metadata_cell_line", "Metadata_seeding_density", "Metadata_time_point"]
).sample(n=50, random_state=0)

umap_obj = UMAP(random_state=0)
umapdf = umap_obj.fit_transform(
    vdf.loc[:, ~vdf.columns.str.contains("metadata", case=False)]
)
umapdf = pd.DataFrame(umapdf, columns=["umap0", "umap1"])
umapdf = umapdf.assign(
    Metadata_cell_line=vdf["Metadata_cell_line"].reset_index(drop=True),
    Metadata_seeding_density=vdf["Metadata_seeding_density"].reset_index(drop=True),
    Metadata_time_point=vdf["Metadata_time_point"].reset_index(drop=True),
)
umapdf["Metadata_seeding_density"] = umapdf["Metadata_seeding_density"].astype(str)
umapdf["Metadata_time_point"] = umapdf["Metadata_time_point"].astype(str)


# In[16]:


# Needed to display in chrome
kaleido.get_chrome_sync()


# In[17]:


fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_cell_line",
    title="UMAP by Cell Line",
    color_discrete_sequence=px.colors.qualitative.Dark24,
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


# In[18]:


blue_green_colors = ["#edf8fb", "#b2e2e2", "#66c2a4", "#2ca25f", "#006d2c"]

fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_seeding_density",
    title="UMAP by Seeding Density",
    color_discrete_sequence=blue_green_colors,
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


# In[19]:


blue_green_short_colors = ["#edf8fb", "#99d8c9", "#2ca25f"]

fig = px.scatter(
    umapdf,
    x="umap0",
    y="umap1",
    color="Metadata_time_point",
    title="UMAP by Time Point",
    color_discrete_sequence=blue_green_short_colors,
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


# # Treemap Cell Count Figures

# In[20]:


cellcountdf = (
    round1df[["Metadata_cell_line", "Metadata_seeding_density", "Metadata_time_point"]]
    .value_counts()
    .reset_index(name="count")
)

fig = px.treemap(
    cellcountdf,
    path=["Metadata_cell_line", "Metadata_seeding_density", "Metadata_time_point"],
    values="count",
    title="Treemap of Cell Counts",
)

fig.show()
fig.write_image(static_figure_path / "round1_cell_count_treemap.png", width=2000, height=1200)
fig.write_html(
    int_figure_path / "round1_cell_count_treemap.html",
    full_html=True,
    include_plotlyjs="embed",
)

