from pathlib import Path

from rltl.utils.utils_os import load_object


def create_latex_tab(key, results):
    sources = {}
    targets = {}

    for d, key in [(sources, "sources"), (targets, "targets")]:
        for id_baseline, mu_sigma in results.items():
            mu = 0
            occ_mu = 0
            for env_name, value in mu_sigma["mu"].items():
                if key in env_name:
                    mu += value
                    occ_mu += 1
            mu /= occ_mu
            sigma = 0
            occ_sigma = 0
            for env_name, value in mu_sigma["sigma"].items():
                if key in env_name:
                    sigma += value
                    occ_sigma += 1
            sigma /= occ_sigma
            d[id_baseline] = (mu, sigma)

    strr = ""
    strr += "&  {} & sources & $ $    &".format(key)
    strr += "& & sources & $\mu$    &"
    strr += (" & ".join(["{:.3f}".format(mu) for id_baseline, (mu, _) in sources.items()]) + "\\\\" + "\n")
    strr += "& &         & $\sigma &"
    strr += (" & ".join(["{:.3f}".format(sigma) for id_baseline, (_, sigma) in sources.items()]) + "\\\\" + "\n")
    strr += "& & targets & $\mu$    &"
    strr += (" & ".join(["{:.3f}".format(mu) for id_baseline, (mu, _) in targets.items()]) + "\\\\" + "\n")
    strr += "& &         & $\sigma$ &"
    strr += (" & ".join(["{:.3f}".format(sigma) for id_baseline, (_, sigma) in targets.items()]) + "\\\\" + "\n")
    return strr


import os

dynamics = {}
baseline_name = None
for file in os.listdir("data"):
    print(file)
    file_path = Path(file)
    dynamics_data_path = "data" / file_path / "0" / "inferences" / "results_dynamics.pickle"
    print(dynamics_data_path)
    if dynamics_data_path.exists():
        dynamics[file] = load_object(dynamics_data_path)
        baseline_name = dynamics[file].keys()

strr = ""
strr += "\\begin{center}\n"
strr += "\\begin{tabular}{ " + " c " * (len(baseline_name) + 4) + "}\n"
strr += "\hline\n"
strr += "& &         &        &"
strr += (" & ".join(["{}".format(id_baseline.replace("_", "\_")) for id_baseline in baseline_name]) + "\\\\" + "\n")
strr += "\hline\n"
for key, item in dynamics.items():
    strr += create_latex_tab(key, item)

strr = strr[:-3] + "\n"
strr += "\\end{tabular}\n"
strr += "\\end{center}"

print(strr)
