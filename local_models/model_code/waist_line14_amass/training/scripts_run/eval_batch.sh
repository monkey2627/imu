cd "$(dirname "${BASH_SOURCE[0]}")/.."
P=/home/duanyuhan/SoftWare/miniconda3/envs/mobileposer/bin/python
export CUDA_VISIBLE_DEVICES=6
ev() { name=$1; poser=$2; joints=$3; shift 3; env "$@" $P -m mobileposer.eval_variant --name $name --poser $poser --joints $joints > eval_$name.log 2>&1; }
ev base      v3_base_poser    v3_base_joints    X=1
ev base_s2   v3_base_s2_poser v3_base_s2_joints X=1
ev biglstm   v3_biglstm_poser v3_biglstm_joints MOBILEPOSER_ARCH=biglstm
ev tf        v3_tf_poser      v3_tf_joints      MOBILEPOSER_ARCH=tf
ev tf250     v3_tf250_poser   v3_tf250_joints   MOBILEPOSER_ARCH=tf
echo done > eval_batch.done
