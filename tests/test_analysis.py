"""
Tests for analysis.ipynb functions.

Assumes the notebook functions have been extracted to a module called `analysis`,
or that the notebook has been converted to a .py file. Adjust the import as needed.

This file lives in tests/ (a subdirectory of the project root). The sys.path
block below adds the project root so that `analysis` can be imported regardless
of how pytest is invoked.

To run from the project root:
    pytest tests/test_analysis.py -v
"""

import sys
from pathlib import Path

# Add the project root (parent of tests/) to sys.path so `analysis` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
import numpy as np
import pandas as pd
import tempfile
import os

# ---------------------------------------------------------------------------
# Adjust this import to match however you've exposed the notebook functions.
# For example, if you've saved the notebook as analysis.py:
#   from analysis import load_data, pre_processing, ...
# ---------------------------------------------------------------------------
from analysis import (
    load_data, pre_processing, drop_duplicates,
    build_design_matrix, build_constrast_matrix,
    map_probes_to_genes, export_for_ipathway, validate_with_targetscan,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def raw_expression_df():
    """
    Minimal raw DataFrame that mimics the shape coming out of load_data()
    before pre_processing — six expression columns plus six matching PVAL columns.
    All p-values are set to 0.01 (significant) so every gene passes the filter.
    """
    np.random.seed(0)
    samples = [
        "miR-neg_rep1", "miR-neg_rep2", "miR-neg_rep3",
        "miR-542-3p_rep1", "miR-542-3p_rep2", "miR-542-3p_rep3",
    ]
    pval_cols = [f"{s}_PVAL" for s in samples]
    n_genes = 20
    expr = pd.DataFrame(
        np.random.uniform(100, 10000, size=(n_genes, len(samples))),
        columns=samples,
        index=[f"PROBE_{i}" for i in range(n_genes)],
    )
    pvals = pd.DataFrame(
        np.full((n_genes, len(pval_cols)), 0.01),
        columns=pval_cols,
        index=expr.index,
    )
    return pd.concat([expr, pvals], axis=1)


@pytest.fixture
def preprocessed_df(raw_expression_df):
    """
    Run pre_processing() on the raw fixture so downstream fixtures can share it.
    Skip (not fail) the test if pre_processing is not importable yet.
    """
    pytest.importorskip("analysis")
    from analysis import pre_processing
    return pre_processing(raw_expression_df)


@pytest.fixture
def expression_df_with_duplicates():
    """
    Small expression DataFrame that contains deliberate duplicate probe IDs.
    PROBE_0 appears twice; the second row has a lower average and should be dropped.
    """
    cols = ["Control_1", "Control_2", "Control_3",
            "Treatment_1", "Treatment_2", "Treatment_3"]
    data = {
        "PROBE_0": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0],   # high mean — keep
        "PROBE_0_dup": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0],  # low mean  — discard
        "PROBE_1": [5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
    }
    df = pd.DataFrame(data, index=cols).T
    # make PROBE_0_dup share the same index label as PROBE_0
    df.index = ["PROBE_0", "PROBE_0", "PROBE_1"]
    return df


@pytest.fixture
def clean_expression_df():
    """
    A small, tidy expression DataFrame (no duplicates, correct column names)
    suitable for feeding into build_design_matrix and analyze_data.
    """
    np.random.seed(42)
    cols = ["Control_1", "Control_2", "Control_3",
            "Treatment_1", "Treatment_2", "Treatment_3"]
    return pd.DataFrame(
        np.random.uniform(5, 15, size=(30, 6)),
        columns=cols,
        index=[f"GENE_{i}" for i in range(30)],
    )


@pytest.fixture
def limma_results_df():
    """
    Fake limma results DataFrame with the columns produced by analyze_data()
    + map_probes_to_genes().
    """
    np.random.seed(7)
    n = 50
    return pd.DataFrame({
        "log2FoldChange": np.random.normal(0, 1.5, n),
        "pvalue": np.random.uniform(0, 1, n),
        "adj_pvalue": np.random.uniform(0, 1, n),
        "Symbol": [f"GENE{i}" for i in range(n)],
    }, index=[f"PROBE_{i}" for i in range(n)])


# ===========================================================================
# load_data
# ===========================================================================

class TestLoadData:
    def test_raises_for_missing_file(self, tmp_path):
        from analysis import load_data
        with pytest.raises(FileNotFoundError):
            load_data(tmp_path / "does_not_exist.txt")

    def test_raises_for_unrecognized_filename(self, tmp_path):
        from analysis import load_data
        f = tmp_path / "some_other_file.txt"
        f.write_text("col1\tcol2\n1\t2\n")
        with pytest.raises(ValueError):
            load_data(f)

    def test_loads_non_normalized_file(self, tmp_path):
        from analysis import load_data
        # Build a minimal file that matches the expected format (4-row header)
        lines = [
            "# comment 1\n", "# comment 2\n", "# comment 3\n", "# comment 4\n",
            "ID_REF\tControl_1\tControl_2\n",
            "PROBE_0\t100\t200\n",
            "PROBE_1\t150\t250\n",
        ]
        f = tmp_path / "GSE47363_non-normalized.txt"
        f.write_text("".join(lines))
        df = load_data(f)
        assert isinstance(df, pd.DataFrame)
        assert "PROBE_0" in df.index
        assert df.shape == (2, 2)

    def test_loads_metadata_file(self, tmp_path):
        from analysis import load_data
        # 8 header rows then a data row
        header = "".join([f"# row {i}\n" for i in range(8)])
        data = "col_a\tcol_b\n1\t2\n"
        f = tmp_path / "metadata_samples.txt"
        f.write_text(header + data)
        df = load_data(f)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["col_a", "col_b"]


# ===========================================================================
# pre_processing
# ===========================================================================

class TestPreProcessing:
    def test_returns_dataframe(self, raw_expression_df):
        from analysis import pre_processing
        result = pre_processing(raw_expression_df)
        assert isinstance(result, pd.DataFrame)

    def test_drops_pval_columns(self, raw_expression_df):
        from analysis import pre_processing
        result = pre_processing(raw_expression_df)
        assert not any("PVAL" in c.upper() for c in result.columns)

    def test_renames_columns(self, raw_expression_df):
        from analysis import pre_processing
        result = pre_processing(raw_expression_df)
        expected = {"Control_1", "Control_2", "Control_3",
                    "Treatment_1", "Treatment_2", "Treatment_3"}
        assert set(result.columns) == expected

    def test_log2_transforms_values(self, raw_expression_df):
        from analysis import pre_processing
        result = pre_processing(raw_expression_df)
        # All raw values are > 100, so log2(x+1) should be > log2(101) ≈ 6.66
        assert (result.values > np.log2(101)).all()

    def test_filters_low_detection_genes(self):
        from analysis import pre_processing
        # Two genes: PROBE_A passes (3 samples with p<=0.05),
        #            PROBE_B fails (only 1 sample with p<=0.05)
        samples = ["miR-neg_rep1", "miR-neg_rep2", "miR-neg_rep3",
                   "miR-542-3p_rep1", "miR-542-3p_rep2", "miR-542-3p_rep3"]
        pval_cols = [f"{s}_PVAL" for s in samples]
        expr = pd.DataFrame(
            {"miR-neg_rep1": [100, 100], "miR-neg_rep2": [100, 100],
             "miR-neg_rep3": [100, 100], "miR-542-3p_rep1": [100, 100],
             "miR-542-3p_rep2": [100, 100], "miR-542-3p_rep3": [100, 100]},
            index=["PROBE_A", "PROBE_B"],
        )
        pvals = pd.DataFrame(
            {c: [0.01, 0.9] for c in pval_cols},
            index=["PROBE_A", "PROBE_B"],
        )
        df = pd.concat([expr, pvals], axis=1)
        result = pre_processing(df)
        assert "PROBE_A" in result.index
        assert "PROBE_B" not in result.index


# ===========================================================================
# drop_duplicates
# ===========================================================================

class TestDropDuplicates:
    def test_returns_no_duplicate_indices(self, expression_df_with_duplicates):
        from analysis import drop_duplicates
        result = drop_duplicates(expression_df_with_duplicates)
        assert result.index.is_unique

    def test_keeps_highest_mean_probe(self, expression_df_with_duplicates):
        from analysis import drop_duplicates
        result = drop_duplicates(expression_df_with_duplicates)
        # PROBE_0 with mean=8 should be kept; mean of retained row should be 8
        assert result.loc["PROBE_0"].mean() == pytest.approx(8.0)

    def test_does_not_add_row_mean_column(self, expression_df_with_duplicates):
        from analysis import drop_duplicates
        result = drop_duplicates(expression_df_with_duplicates)
        assert "row_mean" not in result.columns

    def test_non_duplicate_rows_preserved(self, expression_df_with_duplicates):
        from analysis import drop_duplicates
        result = drop_duplicates(expression_df_with_duplicates)
        assert "PROBE_1" in result.index


# ===========================================================================
# build_design_matrix
# ===========================================================================

class TestBuildDesignMatrix:
    def test_shape_is_samples_by_two(self, clean_expression_df):
        from analysis import build_design_matrix
        dm = build_design_matrix(clean_expression_df)
        assert dm.shape == (6, 2)

    def test_values_are_binary(self, clean_expression_df):
        from analysis import build_design_matrix
        dm = build_design_matrix(clean_expression_df)
        assert set(np.unique(dm)) == {0, 1}

    def test_each_sample_belongs_to_one_group(self, clean_expression_df):
        from analysis import build_design_matrix
        dm = build_design_matrix(clean_expression_df)
        # Each row should sum to 1 (exactly one group)
        assert (np.array(dm).sum(axis=1) == 1).all()

    def test_correct_group_counts(self, clean_expression_df):
        from analysis import build_design_matrix
        dm = build_design_matrix(clean_expression_df)
        # 3 Controls and 3 Treatments
        totals = np.array(dm).sum(axis=0)
        assert sorted(totals) == [3, 3]


# ===========================================================================
# build_constrast_matrix
# ===========================================================================

class TestBuildContrastMatrix:
    def _get_design(self, clean_expression_df):
        from analysis import build_design_matrix
        return build_design_matrix(clean_expression_df)

    def test_returns_dataframe(self, clean_expression_df):
        from analysis import build_constrast_matrix
        dm = self._get_design(clean_expression_df)
        cm = build_constrast_matrix(dm)
        assert isinstance(cm, pd.DataFrame)

    def test_contrast_column_exists(self, clean_expression_df):
        from analysis import build_constrast_matrix
        dm = self._get_design(clean_expression_df)
        cm = build_constrast_matrix(dm)
        assert "Treatment_vs_Control" in cm.columns

    def test_contrast_coefficients_sum_to_zero(self, clean_expression_df):
        from analysis import build_constrast_matrix
        dm = self._get_design(clean_expression_df)
        cm = build_constrast_matrix(dm)
        assert cm["Treatment_vs_Control"].sum() == pytest.approx(0.0)

    def test_contrast_direction(self, clean_expression_df):
        """Treatment coefficient should be +1, Control coefficient should be -1."""
        from analysis import build_constrast_matrix
        dm = self._get_design(clean_expression_df)
        cm = build_constrast_matrix(dm)
        assert 1.0 in cm["Treatment_vs_Control"].values
        assert -1.0 in cm["Treatment_vs_Control"].values


# ===========================================================================
# map_probes_to_genes
# ===========================================================================

class TestMapProbesToGenes:
    def _make_results(self):
        return pd.DataFrame({
            "log2FoldChange": [0.5, -1.2, 2.0],
            "pvalue": [0.01, 0.05, 0.001],
            "adj_pvalue": [0.05, 0.1, 0.01],
        }, index=pd.Index(["P1", "P2", "P3"], name="ID_REF"))

    def _make_annotation_file(self, tmp_path):
        content = (
            "# header line\n"
            "ID\tSymbol\tDefinition\n"
            "P1\tGENE_A\tGene A description\n"
            "P2\tGENE_B\tGene B description\n"
            # P3 intentionally absent — should be dropped
        )
        f = tmp_path / "GPL10558_annot.txt"
        f.write_text(content)
        return f

    def test_raises_for_missing_annotation_file(self, tmp_path):
        from analysis import map_probes_to_genes
        results = self._make_results()
        with pytest.raises(FileNotFoundError):
            map_probes_to_genes(results, tmp_path / "missing.txt")

    def test_symbol_column_present(self, tmp_path):
        from analysis import map_probes_to_genes
        results = self._make_results()
        annot_file = self._make_annotation_file(tmp_path)
        out = map_probes_to_genes(results, annot_file)
        assert "Symbol" in out.columns

    def test_drops_unmapped_probes(self, tmp_path):
        from analysis import map_probes_to_genes
        results = self._make_results()
        annot_file = self._make_annotation_file(tmp_path)
        out = map_probes_to_genes(results, annot_file)
        # P3 has no annotation entry, so it should be absent
        assert "P3" not in out.index

    def test_mapped_probes_present(self, tmp_path):
        from analysis import map_probes_to_genes
        results = self._make_results()
        annot_file = self._make_annotation_file(tmp_path)
        out = map_probes_to_genes(results, annot_file)
        assert "P1" in out.index
        assert out.loc["P1", "Symbol"] == "GENE_A"


# ===========================================================================
# export_for_ipathway
# ===========================================================================

class TestExportForIpathway:
    def test_returns_head_dataframe(self, limma_results_df, tmp_path):
        from analysis import export_for_ipathway
        out = export_for_ipathway(limma_results_df, output_path=tmp_path)
        assert isinstance(out, pd.DataFrame)

    def test_output_columns(self, limma_results_df, tmp_path):
        from analysis import export_for_ipathway
        out = export_for_ipathway(limma_results_df, output_path=tmp_path)
        assert list(out.columns) == ["gene_symbol", "logFC", "adj_pvalue"]

    def test_csv_file_created(self, limma_results_df, tmp_path):
        from analysis import export_for_ipathway
        export_for_ipathway(limma_results_df, filename="test_out.csv", output_path=tmp_path)
        assert (tmp_path / "test_out.csv").exists()

    def test_csv_sorted_by_pvalue(self, limma_results_df, tmp_path):
        from analysis import export_for_ipathway
        export_for_ipathway(limma_results_df, filename="sorted.csv", output_path=tmp_path)
        saved = pd.read_csv(tmp_path / "sorted.csv")
        assert saved["adj_pvalue"].is_monotonic_increasing

    def test_no_file_when_output_path_is_none(self, limma_results_df, tmp_path):
        from analysis import export_for_ipathway
        # Should not raise and should not write a file when output_path=None
        out = export_for_ipathway(limma_results_df, output_path=None)
        assert isinstance(out, pd.DataFrame)


# ===========================================================================
# validate_with_targetscan
# ===========================================================================

class TestValidateWithTargetscan:
    def _make_targetscan_file(self, tmp_path, gene_symbols):
        df = pd.DataFrame({"Gene Symbol": gene_symbols, "Other": range(len(gene_symbols))})
        f = tmp_path / "TargetScan_miR-542-3p.txt"
        df.to_csv(f, sep="\t", index=False)
        return f

    def test_returns_float_pvalue(self, limma_results_df, tmp_path):
        from analysis import validate_with_targetscan
        # Use the first 10 genes as "targets"
        targets = limma_results_df["Symbol"].tolist()[:10]
        ts_file = self._make_targetscan_file(tmp_path, targets)
        ks_p = validate_with_targetscan(limma_results_df, ts_file, output_path=None)
        assert isinstance(ks_p, float)

    def test_pvalue_in_valid_range(self, limma_results_df, tmp_path):
        from analysis import validate_with_targetscan
        targets = limma_results_df["Symbol"].tolist()[:10]
        ts_file = self._make_targetscan_file(tmp_path, targets)
        ks_p = validate_with_targetscan(limma_results_df, ts_file, output_path=None)
        assert 0.0 <= ks_p <= 1.0

    def test_is_target_column_added(self, limma_results_df, tmp_path):
        from analysis import validate_with_targetscan
        targets = limma_results_df["Symbol"].tolist()[:10]
        ts_file = self._make_targetscan_file(tmp_path, targets)
        df_copy = limma_results_df.copy()
        validate_with_targetscan(df_copy, ts_file, output_path=None)
        assert "is_target" in df_copy.columns

    def test_target_genes_correctly_flagged(self, limma_results_df, tmp_path):
        from analysis import validate_with_targetscan
        target_genes = ["GENE0", "GENE1", "GENE2"]
        ts_file = self._make_targetscan_file(tmp_path, target_genes)
        df_copy = limma_results_df.copy()
        validate_with_targetscan(df_copy, ts_file, output_path=None)
        flagged = df_copy[df_copy["is_target"]]["Symbol"].tolist()
        assert set(flagged) == set(target_genes)