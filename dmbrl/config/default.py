from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import importlib.machinery
import importlib.util

import gpflow
from dotmap import DotMap

from dmbrl.modeling.models import NN, BNN, TFGP, GT_dynamics


def create_config(env_name, ctrl_type, ctrl_args, overrides, logdir):
    cfg = DotMap()
    type_map = DotMap(
        exp_cfg=DotMap(
            sim_cfg=DotMap(
                task_hor=int,
                stochastic=make_bool,
                noise_std=float
            ),
            exp_cfg=DotMap(
                ntrain_iters=int,
                nrollouts_per_iter=int,
                ninit_rollouts=int
            ),
            log_cfg=DotMap(
                nrecord=int,
                neval=int
            ),
        ),
        ctrl_cfg=DotMap(
            per=int,
            prop_cfg=DotMap(
                model_pretrained=make_bool,
                npart=int,
                ign_var=make_bool,
            ),
            opt_cfg=DotMap(
                plan_hor=int,
                init_var=float,
            ),
            log_cfg=DotMap(
                save_all_models=make_bool,
                log_traj_preds=make_bool,
                log_particles=make_bool
            ),
            gbp_cfg=DotMap(
                plan_iter=int,
                lr=float,
                gbp_type=int
            ),
            cem_cfg=DotMap(
                cem_type=str,
                training_scheme=str,
                pct_testset=float,
                seed=int,
                policy_network_shape=list,
                policy_epochs=int,
                policy_lr=float,
                policy_weight_decay=float,
                test_policy=int,
                minibatch_size=int,
                training_top_k=int,
                discriminator_network_shape=list,
                discriminator_act_type=str,
                discriminator_norm_type=str,
                gan_type=str,
                discriminator_ent_lambda=float,
                discriminator_lr=float,
                discriminator_epochs=int,
                discriminator_minibatch_size=int,
                discriminator_gradient_penalty_coeff=float,
                zero_weight=str
            ),
            il_cfg=DotMap(
                expert_amc_dir=int,
                use_gt_dynamics=int
            ),
            mb_cfg=DotMap(
                activation=str,
                dynamics_lr=float,
                normalization=str,
                do_benchmarking=str,
                mb_batch_size=int,
                mb_epochs=int
            ),
        )
    )

    dir_path = os.path.dirname(os.path.realpath(__file__))
    loader = importlib.machinery.SourceFileLoader(
        env_name, os.path.join(dir_path, "%s.py" % env_name)
    )
    spec = importlib.util.spec_from_loader(loader.name, loader)
    cfg_source = importlib.util.module_from_spec(spec)
    loader.exec_module(cfg_source)
    cfg_module = cfg_source.CONFIG_MODULE()

    _create_exp_config(cfg.exp_cfg, cfg_module, logdir, type_map)
    _create_ctrl_config(cfg.ctrl_cfg, cfg_module, ctrl_type, ctrl_args, type_map)

    for (k, v) in overrides:
        apply_override(cfg, type_map, k, v)

    # needed for GAN training
    cfg.ctrl_cfg.cem_cfg.init_var = cfg.ctrl_cfg.opt_cfg.init_var

    # needed for mb_benchmarking. should not be used anyway else
    cfg.ctrl_cfg.opt_cfg.mb_cfg = cfg.ctrl_cfg.mb_cfg

    return cfg


def _create_gbp_config(exp_cfg):
    exp_cfg.plan_iter = 5
    exp_cfg.lr = 0.03
    exp_cfg.gbp_type = 3


def _create_il_config(exp_cfg):
    default_path = [
        # desktop
        "/home/tingwu/imitation-rl/data/humanoid_mocap/all_asfamc/test",
        # vector cluster
        "/scratch/gobi2/tingwu/data/humanoid_mocap/all_asfamc/test"
    ]
    exp_cfg.expert_amc_dir = None
    for path in default_path:
        if os.path.exists(path):
            exp_cfg.expert_amc_dir = path

    exp_cfg.use_gt_dynamics = 0


def _create_mb_config(exp_cfg):
    exp_cfg.activation = 'swish'
    exp_cfg.dynamics_lr = 0.001
    exp_cfg.normalization = 'none'
    exp_cfg.mb_batch_size = 32
    exp_cfg.mb_epochs = 100
    exp_cfg.do_benchmarking = 'no'  # super hacky


def _create_cem_config(exp_cfg):
    exp_cfg.cem_type = 'POPLINA-INIT'
    exp_cfg.training_scheme = 'BC-AR'

    # the policy network
    exp_cfg.pct_testset = 0.2  # 20% for the test set
    exp_cfg.seed = 1234
    exp_cfg.policy_network_shape = [64, 64]
    exp_cfg.policy_epochs = 100
    exp_cfg.policy_lr = 3e-3
    exp_cfg.minibatch_size = 64
    exp_cfg.policy_weight_decay = 1e-5  # test this 1e-5
    exp_cfg.test_policy = 0
    exp_cfg.zero_weight = 'No'

    # the discriminator
    exp_cfg.discriminator_network_shape = [64, 64]
    exp_cfg.discriminator_act_type = 'leaky_relu'
    exp_cfg.discriminator_norm_type = None
    exp_cfg.gan_type = 'GAN'
    exp_cfg.discriminator_ent_lambda = 1e-3
    exp_cfg.discriminator_lr = 3e-3
    exp_cfg.discriminator_epochs = 40  # test this
    exp_cfg.discriminator_minibatch_size = 64
    exp_cfg.discriminator_gradient_penalty_coeff = 10.0

    # policy distilation
    exp_cfg.training_top_k = 50
    exp_cfg.pwcem_init_mean = True


def _create_exp_config(exp_cfg, cfg_module, logdir, type_map):
    exp_cfg.sim_cfg.env = cfg_module.ENV
    exp_cfg.sim_cfg.task_hor = cfg_module.TASK_HORIZON

    exp_cfg.exp_cfg.ntrain_iters = cfg_module.NTRAIN_ITERS
    exp_cfg.exp_cfg.nrollouts_per_iter = cfg_module.NROLLOUTS_PER_ITER

    exp_cfg.log_cfg.logdir = logdir


def _create_ctrl_config(ctrl_cfg, cfg_module, ctrl_type, ctrl_args, type_map):
    """Creates controller configuration.

    """
    if ctrl_type == "MPC":
        ctrl_cfg.env = cfg_module.ENV
        if hasattr(cfg_module, "UPDATE_FNS"):
            ctrl_cfg.update_fns = cfg_module.UPDATE_FNS
        if hasattr(cfg_module, "obs_preproc"):
            ctrl_cfg.prop_cfg.obs_preproc = cfg_module.obs_preproc
        if hasattr(cfg_module, "obs_postproc"):
            ctrl_cfg.prop_cfg.obs_postproc = cfg_module.obs_postproc
        if hasattr(cfg_module, "obs_postproc2"):
            ctrl_cfg.prop_cfg.obs_postproc2 = cfg_module.obs_postproc2
        if hasattr(cfg_module, "targ_proc"):
            ctrl_cfg.prop_cfg.targ_proc = cfg_module.targ_proc

        ctrl_cfg.opt_cfg.plan_hor = cfg_module.PLAN_HOR
        ctrl_cfg.opt_cfg.init_var = cfg_module.INIT_VAR
        ctrl_cfg.opt_cfg.obs_cost_fn = cfg_module.obs_cost_fn
        ctrl_cfg.opt_cfg.ac_cost_fn = cfg_module.ac_cost_fn
        if hasattr(cfg_module, "obs_ac_cost_fn"):
            ctrl_cfg.prop_cfg.obs_ac_cost_fn = cfg_module.obs_ac_cost_fn
        else:
            ctrl_cfg.prop_cfg.obs_ac_cost_fn = None

        # Process arguments here.
        model_init_cfg = ctrl_cfg.prop_cfg.model_init_cfg
        if ctrl_args.get("model-type", "PE") in ["GT"]:
            ctrl_args["model-type"] = ctrl_args.get("model-type", "PE")
            model_init_cfg.model_class = GT_dynamics
            type_map.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = 1

            ctrl_cfg.prop_cfg.model_train_cfg = cfg_module.NN_TRAIN_CFG
            model_init_cfg.model_constructor = GT_dynamics.none_constructor

            # Add possible overrides
            type_map.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = str
            type_map.ctrl_cfg.prop_cfg.model_init_cfg.load_model = make_bool
            type_map.ctrl_cfg.prop_cfg.model_train_cfg = DotMap(
                batch_size=int, epochs=int,
                holdout_ratio=float, max_logging=int
            )

        elif ctrl_args.get("model-type", "PE") in ["P", "PE", "D", "DE"]:
            ctrl_args["model-type"] = ctrl_args.get("model-type", "PE")
            if ctrl_args["model-type"][0] == "P":
                model_init_cfg.model_class = BNN
            else:
                model_init_cfg.model_class = NN

            if len(ctrl_args["model-type"]) == 1:
                model_init_cfg.num_nets = 1
                type_map.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = \
                    create_read_only("Number of nets for non-ensembled nets must be one, do not modify.")
            else:
                model_init_cfg.num_nets = 5
                type_map.ctrl_cfg.prop_cfg.model_init_cfg.num_nets = create_conditional(
                    int, lambda string: int(string) > 1, "Ensembled models must have more than one net."
                )
            ctrl_cfg.prop_cfg.model_train_cfg = cfg_module.NN_TRAIN_CFG
            model_init_cfg.model_constructor = cfg_module.nn_constructor

            # Add possible overrides
            type_map.ctrl_cfg.prop_cfg.model_init_cfg.model_dir = str
            type_map.ctrl_cfg.prop_cfg.model_init_cfg.load_model = make_bool

            type_map.ctrl_cfg.prop_cfg.model_train_cfg = DotMap(
                batch_size=int, epochs=int,
                holdout_ratio=float, max_logging=int
            )
        elif ctrl_args["model-type"] == "GP":
            model_init_cfg.model_class = TFGP
            model_init_cfg.kernel_class = gpflow.kernels.RBF
            model_init_cfg.kernel_args = {
                "input_dim": cfg_module.MODEL_IN,
                "output_dim": cfg_module.MODEL_OUT,
                "ARD": True
            }
            model_init_cfg.num_inducing_points = cfg_module.GP_NINDUCING_POINTS
            model_init_cfg.model_constructor = cfg_module.gp_constructor
        else:
            raise NotImplementedError("Unknown model type.")

        ctrl_cfg.prop_cfg.mode = ctrl_args.get("prop-type", "TSinf")
        ctrl_cfg.prop_cfg.npart = 20
        # Handle special cases
        if ctrl_cfg.prop_cfg.mode[:2] == "TS":
            if ctrl_args["model-type"] not in ["PE", "DE"]:
                raise RuntimeError("Cannot perform TS with non-ensembled models.")
            if ctrl_args["model-type"] == "DE":
                ctrl_cfg.prop_cfg.ign_var = True
                type_map.ctrl_cfg.prop_cfg.ign_var = \
                    create_read_only("DE-TS* methods must ignore variance, do not modify.")
        if ctrl_cfg.prop_cfg.mode == "E":
            ctrl_cfg.prop_cfg.npart = 1
            type_map.ctrl_cfg.prop_cfg.npart = \
                create_read_only("Only need one particle for deterministic propagation, do not modify.")
        if ctrl_args["model-type"] == "D" and ctrl_cfg.prop_cfg.mode != "E":
            raise ValueError("Can only use deterministic propagation for deterministic models.")

        ctrl_cfg.opt_cfg.mode = ctrl_args.get("opt-type", "CEM")
        if ctrl_cfg.opt_cfg.mode == "CEM":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                max_iters=int,
                popsize=int,
                num_elites=int,
                epsilon=float,
                alpha=float
            )
        elif ctrl_cfg.opt_cfg.mode == "POPLIN-A":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                max_iters=int,
                popsize=int,
                num_elites=int,
                epsilon=float,
                alpha=float
            )
        elif ctrl_cfg.opt_cfg.mode == "POPLIN-P":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                max_iters=int,
                popsize=int,
                num_elites=int,
                epsilon=float,
                alpha=float
            )
        elif ctrl_cfg.opt_cfg.mode == "GBPCEM":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                max_iters=int,
                popsize=int,
                num_elites=int,
                epsilon=float,
                alpha=float
            )
        elif ctrl_cfg.opt_cfg.mode == "Random":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                popsize=int
            )
        elif ctrl_cfg.opt_cfg.mode == "GBPRandom":
            type_map.ctrl_cfg.opt_cfg.cfg = DotMap(
                popsize=int
            )
        else:
            raise NotImplementedError("Unknown optimizer.")
        ctrl_cfg.opt_cfg.cfg = cfg_module.OPT_CFG[ctrl_cfg.opt_cfg.mode]
    else:
        raise NotImplementedError("Unknown controller class.")

    # the config added to control the gradient based planning
    _create_gbp_config(ctrl_cfg.gbp_cfg)
    _create_cem_config(ctrl_cfg.cem_cfg)
    _create_il_config(ctrl_cfg.il_cfg)
    _create_mb_config(ctrl_cfg.mb_cfg)


def apply_override(cfg, type_map, override_key, value, prefix=''):
    """Modifies the configuration to apply the given override.
    """
    pth = override_key.split(".")
    filter_pth = prefix.split(".")
    # hack for lists
    if value.startswith('[') and value.endswith(']'):
        value = value.replace('[', '')
        value = value.replace(']', '')
        value = value.split(',')
        value = [int(val) for val in value if val != '']
    if len(prefix) == 0 or pth[:len(filter_pth)] == prefix.split("."):
        cur_map = cfg
        cur_type_map = type_map
        try:
            for key in pth[:-1]:
                cur_map = cur_map[key]
                cur_type_map = cur_type_map[key]
        except KeyError:
            raise KeyError(
                "Either %s cannot be overridden (is a function/object/class/etc.) or "
                "the type map is not updated." % override_key
            )
        if cur_type_map.get(pth[-1], None) is None:
            raise KeyError(
                "Either %s cannot be overridden (is a function/object/class/etc.) or "
                "the type map is not updated." % override_key
            )
        cur_map[pth[-1]] = cur_type_map[pth[-1]](value)


def make_bool(arg):
    if arg == "False" or arg == "false" or not bool(arg):
        return False
    else:
        return True


def create_read_only(message):
    def read_only(arg):
        raise RuntimeError(message)
    return read_only


def create_conditional(cl, cond, message):
    def conditional(arg):
        if cond(arg):
            return cl(arg)
        else:
            raise RuntimeError(message)
    return conditional
