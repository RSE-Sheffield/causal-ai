-- causal_ai/0.1.0 modulefile for Bede (aarch64)
-- Usage: module use /nobackup/projects/bddur53/causal-ai/modulefiles
--        module load causal_ai/0.1.0

help([[
causal-ai v0.1.0 - Causal inference for AI/ML workflows enabling AI for science

After loading, run:
    python -m causal_ai --help

Documentation: see /nobackup/projects/bddur53/causal-ai/README.md
]])

whatis("Name:        causal-ai")
whatis("Version:     0.1.0")
whatis("Description: Causal inference for AI/ML workflows enabling AI for science")

-- Paths on Bede
local project_root = "/nobackup/projects/bddur53/causal-ai"
local conda_base   = "/nobackup/projects/bddur53/cs1fxa/Miniforge"
local conda_env    = conda_base .. "/envs/ai-4-science"

-- Prevent loading multiple versions
conflict("causal_ai")

-- Conda environment activation
setenv("CONDA_DEFAULT_ENV", "ai-4-science")
setenv("CONDA_PREFIX", conda_env)
setenv("CONDA_SHLVL", "1")

-- Put conda env bins first on PATH
prepend_path("PATH", conda_env .. "/bin")

-- Make the causal_ai package importable
prepend_path("PYTHONPATH", project_root)

-- Convenience variable pointing to the project
setenv("CAUSAL_AI_HOME", project_root)

if mode() == "load" then
    LmodMessage("Loaded causal-ai v0.1.0 (conda env: ai-4-science)")
elseif mode() == "unload" then
    LmodMessage("Unloaded causal-ai v0.1.0")
end
