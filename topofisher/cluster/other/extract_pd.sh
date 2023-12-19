

find /projects/0/gusr0688/pd_sancho -type f -name "*_100.tar.gz" -exec tar -xzvf {} -C /projects/0/gusr0688/vk/pd_sancho \;


sbatch $topofisher_dir/cluster/vectorization/vectorizePD.sh "$vk_dir"/pd_sancho "$vk_dir"/outputs/cluster histogram 0 0 500 cluster

$topofisher_dir/cluster/vectorization/vectorizePD.sh "$vk_dir"/pd_sancho outputs/bash_run histogram 0 0 2 cluster