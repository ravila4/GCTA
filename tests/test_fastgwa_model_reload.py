"""Synthetic CLI regression tests; run with GCTA=/path/to/gcta64 python3 this_file.py -v."""

import csv
import ctypes
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest


class ModelHeader(ctypes.LittleEndianStructure):
    _fields_ = [
        ("magic", ctypes.c_char * 5),
        ("num_indi", ctypes.c_uint32),
        ("fam_flag", ctypes.c_bool),
        ("isGrammar", ctypes.c_bool),
    ]


class FastGWAModelReload(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        binary = os.environ.get("GCTA", "gcta64")
        cls.binary = shutil.which(binary)
        if cls.binary is None:
            raise RuntimeError("Set GCTA to the gcta64 executable to test")
        cls.temporary = tempfile.TemporaryDirectory(prefix="gcta-model-reload-")
        cls.root = Path(cls.temporary.name)
        cls.n = 240
        cls.m = 2048  # GRAMMAR tuning requires at least 2,000 autosomal markers.
        rng = random.Random(20261109)
        cls.genotypes = []
        for _ in range(cls.m):
            frequency = rng.uniform(0.05, 0.45)
            cls.genotypes.append([
                int(rng.random() < frequency) + int(rng.random() < frequency)
                for _ in range(cls.n)
            ])
        order = list(range(cls.n))
        cls.write_genotypes("original", order)
        rng.shuffle(order)
        cls.write_genotypes("permuted", order)
        family_effect = [rng.gauss(0, 2) for _ in range(cls.n // 4)]
        with (cls.root / "phenotype.txt").open("w") as phenotype, \
                (cls.root / "covariates.txt").open("w") as covariates, \
                (cls.root / "related.grm.id").open("w") as ids, \
                (cls.root / "related.grm.sp").open("w") as grm:
            for sample in range(cls.n):
                first = rng.gauss(0, 1) + cls.genotypes[0][sample]
                second = rng.gauss(0, 1)
                value = (1.4 * first - 0.7 * second
                         + family_effect[sample // 4] + rng.gauss(0, 0.3))
                identity = f"f{sample:03d} i{sample:03d}"
                phenotype.write(f"{identity} {value:.17g}\n")
                covariates.write(f"{identity} {first:.17g} {second:.17g}\n")
                ids.write(identity + "\n")
                for other in range(sample // 4 * 4, sample + 1):
                    relatedness = 1 if sample == other else 0.5
                    grm.write(f"{sample + 1} {other + 1} {relatedness}\n")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    @classmethod
    def write_genotypes(cls, name, order):
        prefix = cls.root / name
        with prefix.with_suffix(".fam").open("w") as fam:
            for sample in order:
                fam.write(f"f{sample:03d} i{sample:03d} 0 0 1 -9\n")
        with prefix.with_suffix(".bim").open("w") as bim:
            for marker in range(cls.m):
                bim.write(f"21 rs{marker:04d} 0 {marker + 1} A G\n")
        # SNP-major PLINK BED: 00=A1/A1, 10=A1/A2, 11=A2/A2.
        codes = (3, 2, 0)
        with prefix.with_suffix(".bed").open("wb") as bed:
            bed.write(bytes((0x6C, 0x1B, 0x01)))
            for genotype in cls.genotypes:
                for start in range(0, cls.n, 4):
                    packed = sum(codes[genotype[sample]] << (2 * offset)
                                 for offset, sample in enumerate(order[start:start + 4]))
                    bed.write(bytes((packed,)))

    def run_gcta(self, name, *arguments):
        output = self.root / name
        command = [self.binary, *map(str, arguments), "--thread-num", "1",
                   "--nofilter", "--out", str(output)]
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1")
        result = subprocess.run(command, cwd=self.root, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout[-6000:])
        return output

    def read_results(self, prefix):
        with Path(str(prefix) + ".fastGWA").open() as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(len(rows), self.m)
        self.assertEqual(len({row["SNP"] for row in rows}), self.m)
        return {row["SNP"]: row for row in rows}

    def assert_results_match(self, expected, observed):
        self.assertEqual(expected.keys(), observed.keys())
        for marker, reference in expected.items():
            actual = observed[marker]
            for field in ("N", "A1", "A2"):
                self.assertEqual(reference[field], actual[field], (marker, field))
            for field in ("AF1", "BETA", "SE", "P"):
                left, right = float(reference[field]), float(actual[field])
                self.assertTrue(math.isfinite(left) and math.isfinite(right),
                                (marker, field, left, right))
                # GCTA writes six significant digits; permit rounding at that precision.
                self.assertTrue(math.isclose(left, right, rel_tol=2e-5, abs_tol=1e-10),
                                (marker, field, left, right))

    def check_reload(self, method, joint=True):
        name = method.removeprefix("--fastGWA-") + ("-joint" if joint else "-default")
        arguments = ["--bfile", self.root / "original", "--pheno",
                     self.root / "phenotype.txt", "--qcovar",
                     self.root / "covariates.txt", method, "--seed", "719"]
        if joint:
            arguments += ["--joint-covar"]
        if method != "--fastGWA-lr":
            arguments += ["--grm-sparse", self.root / "related"]
        fused = self.run_gcta(name + "-fused", *arguments)
        fit = self.run_gcta(name + "-fit", *arguments, "--model-only")
        header = ModelHeader.from_buffer_copy(
            Path(str(fit) + ".fastGWA.mdl.bin").read_bytes()[:ctypes.sizeof(ModelHeader)])
        self.assertEqual(header.num_indi, self.n)
        self.assertEqual(header.fam_flag, method != "--fastGWA-lr",
                         "The mixed-model control must not fall back to linear regression")
        if method != "--fastGWA-lr":
            self.assertEqual(header.isGrammar, method == "--fastGWA-mlm")
        expected = self.read_results(fused)
        for genotype in ("original", "permuted"):
            with self.subTest(genotype_order=genotype):
                loaded = self.run_gcta(name + "-loaded-" + genotype,
                                       "--bfile", self.root / genotype,
                                       "--load-model", str(fit) + ".fastGWA")
                self.assert_results_match(expected, self.read_results(loaded))

    def test_linear_joint_covariates(self):
        self.check_reload("--fastGWA-lr")

    def test_mixed_grammar_joint_covariates(self):
        self.check_reload("--fastGWA-mlm")

    def test_mixed_exact_joint_covariates(self):
        self.check_reload("--fastGWA-mlm-exact")

    def test_linear_default_covariates(self):
        self.check_reload("--fastGWA-lr", joint=False)


if __name__ == "__main__":
    unittest.main()
