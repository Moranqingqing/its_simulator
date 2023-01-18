gan_baselines=(
#  "agglo_gan"
#  "hyper_gan"
#  "hyper_nn"
  "hyper_nn_no_dropout"
  "hyper_gan_no_dropout"
)



for i in "${gan_baselines[@]}"; do
  PYTHONPATH="$PYTHONPATH:$4" "$3" main.py $1 --jobs learn_gans --gan_baselines $i --show_plots False --override_config_file $2 &
  pids[${i}]=$!
  k=$((k + 1))
done