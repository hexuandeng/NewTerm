import json
import mlcroissant as mlc

ds = mlc.Dataset("https://raw.githubusercontent.com/mlcommons/croissant/main/datasets/1.0/gpt-3/metadata.json")
gpt = ds.metadata

# FileObjects and FileSets define the resources of the dataset.
distribution = [
    # gpt-3 is hosted on a GitHub repository:
    mlc.FileObject(
        id="github-repository",
        name="github-repository",
        description="NewTerm repository on GitHub.",
        content_url="https://github.com/hexuandeng/NewTerm",
        encoding_format="git+https",
        sha256="main",
    ),
    # Within that repository, a FileSet lists all JSONL files:
    mlc.FileSet(
        id="jsonl-files",
        name="jsonl-files",
        description="JSONL files are hosted on the GitHub repository.",
        contained_in=["github-repository"],
        encoding_format="application/jsonlines",
        includes="benchmark_2022/*.json",
    ),
]
record_sets = [
    # RecordSets contains records in the dataset.
    mlc.RecordSet(
        id="jsonl",
        name="jsonl",
        # Each record has one or many fields...
        fields=[
            # Fields can be extracted from the FileObjects/FileSets.
            mlc.Field(
                id="jsonl/term",
                name="term",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="term"),
                ),
            ),
            mlc.Field(
                id="jsonl/meaning",
                name="meaning",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="meaning"),
                ),
            ),
            mlc.Field(
                id="jsonl/type",
                name="type",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="type"),
                ),
            ),
            mlc.Field(
                id="jsonl/question",
                name="question",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="question"),
                ),
            ),
            mlc.Field(
                id="jsonl/choices",
                name="choices",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="choices"),
                ),
            ),
            mlc.Field(
                id="jsonl/gold",
                name="gold",
                description="",
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    # Extract the field from the column of a FileObject/FileSet:
                    extract=mlc.Extract(column="gold"),
                ),
            ),
            mlc.Field(
                id="jsonl/task",
                name="task",
                description=(
                    "The machine learning task appearing as the name of the"
                    " file."
                ),
                data_types=mlc.DataType.TEXT,
                source=mlc.Source(
                    file_set="jsonl-files",
                    extract=mlc.Extract(
                        file_property=mlc._src.structure_graph.nodes.source.FileProperty.filename
                    ),
                    # Extract the field from a regex on the filename:
                    transforms=[mlc.Transform(regex="^(.*)\.jsonl$")],
                ),
            ),
        ],
    )
]

# Metadata contains information about the dataset.
metadata = mlc.Metadata(
    name="NewTerm",
    # Descriptions can contain plain text or markdown.
    description=(
        ""
    ),
    cite_as=(
    ),
    in_language='en',
    url="https://github.com/hexuandeng/NewTerm",
    distribution=distribution,
    record_sets=record_sets,
    license='https://creativecommons.org/licenses/by-sa/4.0/'
)

with open("croissant.json", "w") as f:
    content = metadata.to_json()
    content = json.dumps(content, indent=2)
    print(content)
    f.write(content)
    f.write("\n")  # Terminate file with newline

dataset = mlc.Dataset(jsonld="croissant.json", debug=True)
records = dataset.records(record_set="jsonl")

for i, record in enumerate(records):
    print(record)
    if i > 10:
        break
print(1)

# print(f"{metadata['name']}: {metadata['description']}")
# for x in ds.records(record_set="default"):
#     print(x)
# newterm = mlc.Metadata()
# newterm.description = "An adaptive benchmark, NewTerm, for real-time evaluation of new terms."
# newterm.
