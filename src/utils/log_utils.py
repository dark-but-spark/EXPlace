"""
Log-related helper functions for training metrics, runtime and best metrics CSV.
"""
import os
import csv


def save_runtime(path, episode, state_transition_time, training_time):
    """Append one row to runtime.csv. Call when episode_counter % 10 == 0."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("episode,state_transition_time,training_time\n")
    with open(path, 'a') as f:
        f.write(f"{episode},{state_transition_time:.4f},{training_time:.4f}\n")


def save_physical_evaluation(
    path, episode, kind, gp_hpwl, tns, wns, valid=True, failure_reason=""
):
    """Record physical results and whether the final placement is usable.

    Invalid DREAMPlace output is retained for diagnosis, but callers must not
    use it as a best result or adaptive-reward observation.
    """
    fieldnames = [
        'episode', 'kind', 'valid', 'failure_reason',
        'gp_hpwl', 'tns', 'wns',
    ]
    file_exists = os.path.exists(path)
    if file_exists:
        # A failed task may be resumed from a run written before the validity
        # columns existed. Migrate that header once before appending, keeping
        # old observations as valid because they had no rejection signal.
        with open(path, newline='') as handle:
            reader = csv.DictReader(handle)
            old_rows = list(reader)
            old_fields = reader.fieldnames or []
        if old_fields != fieldnames:
            with open(path, 'w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for old_row in old_rows:
                    row = {name: old_row.get(name, '') for name in fieldnames}
                    row['valid'] = old_row.get('valid', '1') or '1'
                    row['failure_reason'] = old_row.get('failure_reason', '')
                    writer.writerow(row)
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'episode': int(episode),
            'kind': kind,
            'valid': int(bool(valid)),
            'failure_reason': failure_reason,
            'gp_hpwl': float(gp_hpwl),
            'tns': float(tns),
            'wns': float(wns),
        })


def save_best_metrics(path, episode, metric_type, tns, gp_hpwl, wns, reward_info, actor_loss, critic_loss):
    """Append one row when best TNS or best GP-HPWL is updated. metric_type is 'tns' or 'gp_hpwl'."""
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write("episode,metric_type,tns,wns,gp_hpwl,scaled_reward,hpwl,hierarchy_cost,regularity,displacement_cost,port_cost,df_cost,actor_loss,critic_loss\n")
    r = reward_info or {}
    with open(path, 'a') as f:
        f.write(f"{episode},{metric_type},{tns},{wns},{gp_hpwl},{r.get('scaled_reward', '')},{r.get('wire', '')},{r.get('hier', '')},{r.get('reg', '')},{r.get('displacement', '')},{r.get('port', '')},{r.get('df', '')},{actor_loss},{critic_loss}\n")


def save_eval_metrics(path, writer, episode, hpwl, hierarchy_cost, regularity, displacement_cost, port_cost, df_cost, scaled_reward, tns, wns, gp_hpwl, actor_loss, critic_loss):
    """Append one row to eval_metrics.csv and log scalars to TensorBoard writer."""
    if not os.path.exists(path):
        with open(path, 'a') as f:
            f.write(f"episode,scaled_reward,hpwl,hierarchy_cost,regularity,displacement_cost,port_cost,df_cost,tns,wns,gp_hpwl,actor_loss,critic_loss\n")
    with open(path, 'a') as f:
        f.write(f'{episode},{scaled_reward},{hpwl},{hierarchy_cost},{regularity},{displacement_cost},{port_cost},{df_cost},{tns},{wns},{gp_hpwl},{actor_loss},{critic_loss}\n')

    # TensorBoard logging
    writer.add_scalar('loss/total_reward', scaled_reward, episode)
    writer.add_scalar('cost/hpwl', hpwl, episode)
    writer.add_scalar('cost/hier', hierarchy_cost, episode)
    writer.add_scalar('cost/reg', regularity, episode)
    writer.add_scalar('cost/displacement', displacement_cost, episode)
    writer.add_scalar('cost/port', port_cost, episode)
    writer.add_scalar('cost/df', df_cost, episode)
    writer.add_scalar('cost/tns', tns, episode)
    writer.add_scalar('cost/wns', wns, episode)
    if gp_hpwl != 0:
        writer.add_scalar('cost/gp_hpwl', gp_hpwl, episode)
        writer.add_scalar('loss/actor_loss', actor_loss, episode)
        writer.add_scalar('loss/critic_loss', critic_loss, episode)


def save_reward_weights(path, episode, weights, ppa_penalty='', correlations=None, updated=False):
    """Append one adaptive-reward observation in an analysis-friendly format."""
    correlations = correlations or {}
    mask_names = list(weights.keys())
    fieldnames = (
        ['episode', 'ppa_penalty', 'updated']
        + [f'weight_{name}' for name in mask_names]
        + [f'corr_{name}' for name in mask_names]
    )
    row = {'episode': episode, 'ppa_penalty': ppa_penalty, 'updated': int(updated)}
    row.update({f'weight_{name}': value for name, value in weights.items()})
    row.update({f'corr_{name}': value for name, value in correlations.items()})
    file_exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
