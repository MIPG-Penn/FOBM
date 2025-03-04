tmux new-session -s ses-0 -n FOBM -d

N=5



RUNCMD="python fobm.py -n 1000 -f 10 -t 70"


for (( i=0; i < $N; ++i ))
do
	tmux split-window -h 
	tmux select-layout tiled
	tmux send-keys -t $i "$RUNCMD" C-m
done
tmux send-keys -t $N "htop" C-m

tmux attach-session -t ses-0
