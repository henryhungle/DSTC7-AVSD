
# input 
stage=$1
word_emb_name=$2 # e.g. glove.6B.200d.txt
embed_size=$3
num_epochs=$4

pretrained_elmo=0
elmo_num_outputs=1
finetune_elmo=0
pretrained_all=1
concat_his=1
add_word_emb=1

. path.sh

workdir=`pwd`
data_root=../../../data/dstc7
train_set=$data_root/train_set4DSTC7-AVSD.json
valid_set=$data_root/valid_set4DSTC7-AVSD.json
test_set=$data_root/test_set.json

# directory to read feature files
fea_dir=$data_root
# feature file pattern
fea_file="<FeaType>/<ImageID>.npy"
# input feature types
fea_type="i3d_rgb vggish"

# network architecture
# multimodal encoder
enc_psize="512 512"   # dims of projection layers for input features
enc_hsize="0 0"       # dims of cell states (0: no LSTM layer)
att_size=128          # dim to decide temporal attention
mout_size=512         # dim of final projection layer
# input (question) encoder 
in_enc_layers=1 
in_enc_hsize=512 
# hierarchical history encoder
hist_enc_layers="1 1"  # numbers of word-level layers & QA-pair layers
hist_enc_hsize=512     # dim of hidden layer
hist_out_size=512      # dim of final projection layer
# response (answer) decoder
dec_layers=1    # number of layers
dec_psize=256   # dim of word-embedding layer
dec_hsize=512   # dim of cell states

# training params
batch_size=64   # batch size
max_length=256  # batch size is reduced if len(input_feature) >= max_length
optimizer=Adam  # ADAM optimizer 
seed=1          # random seed

# generator params
beam=5            # beam width
penalty=1.0       # penalty added to the score of each hypothesis
nbest=5           # number of hypotheses to be output
model_epoch=best  # model epoch number to be used

. utils/parse_options.sh || exit 1;

# directory and feature file setting
enc_psize_=`echo $enc_psize|sed "s/ /-/g"`
enc_hsize_=`echo $enc_hsize|sed "s/ /-/g"`
fea_type_=`echo $fea_type|sed "s/ /-/g"`

q_att='conv_sum'
rnn_type="gru" 
mm_att="conv"
ft_fusioning='baseline'
mm_fusioning='nonlinear_multiply'
classifier='logit' 
caption=true
sep_caption=true 
c_att='c_conv_sum'

expdir=new_exps/final4_avsd_${fea_type_}_pretrainedemb${word_emb_name}

# preparation
if [ $stage -le 1 ]; then
    echo -------------------------
    echo stage 1: preparation 
    echo -------------------------
    echo setup ms-coco evaluation tool
    if [ ! -d utils/coco-caption ]; then
        git clone https://github.com/tylin/coco-caption utils/coco-caption
        patch -p0 -u < utils/coco-caption.patch
    else
        echo Already exists.
    fi
    echo -------------------------
    echo checking feature files in $fea_dir
    for ftype in $fea_type; do
        if [ ! -d $fea_dir/$ftype ]; then
            echo cannot access: $fea_dir/$ftype
            echo download and extract feature files into the directory
            exit
        fi
        echo ${ftype}: `ls $fea_dir/$ftype | wc -l`
    done
fi

# training phase
mkdir -p $expdir
if [ $stage -le 2 ]; then
    echo -------------------------
    echo stage 2: model training
    echo -------------------------
    python train.py \
      --gpu 0 \
      --optimizer $optimizer \
      --fea-type $fea_type \
      --train-path "$fea_dir/$fea_file" \
      --train-set $train_set \
      --valid-path "$fea_dir/$fea_file" \
      --valid-set $valid_set \
      --num-epochs $num_epochs \
      --batch-size $batch_size \
      --max-length $max_length \
      --model $expdir/avsd_model \
      --enc-psize $enc_psize \
      --enc-hsize $enc_hsize \
      --att-size $att_size \
      --mout-size $mout_size \
      --embed-size $embed_size \
      --in-enc-layers $in_enc_layers \
      --in-enc-hsize $in_enc_hsize \
      --hist-enc-layers $hist_enc_layers \
      --hist-enc-hsize $hist_enc_hsize \
      --hist-out-size $hist_out_size \
      --dec-layers $dec_layers \
      --dec-psize $dec_psize \
      --dec-hsize $dec_hsize \
      --rand-seed $seed \
      --q-att $q_att \
      --classifier $classifier \
      --rnn-type $rnn_type \
      --mm-att $mm_att \
      --mm-fusioning $mm_fusioning \
      --lr-scheduler \
      --pretrained-word-emb ${data_root}/${word_emb_name} \
      --pretrained-elmo ${pretrained_elmo} \
      --elmo-num-outputs ${elmo_num_outputs} \
      --finetune-elmo ${finetune_elmo} \
      --add-word-emb ${add_word_emb} \
      --pretrained-all ${pretrained_all} \
      --concat-his ${concat_his}
fi

# testing phase
if [ $stage -le 3 ]; then
    echo -----------------------------
    echo stage 3: generate responses
    echo -----------------------------
    for data_set in $test_set; do
        echo start response generation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_b${beam}_p${penalty}.json
        test_log=${result%.*}.log
        python generate.py \
          --gpu 0 \
          --test-path "$fea_dir/$fea_file" \
          --test-set $data_set \
          --model-conf $expdir/avsd_model.conf \
          --model $expdir/avsd_model_${model_epoch} \
          --beam $beam \
          --penalty $penalty \
          --nbest $nbest \
          --output $result 
    done
fi

# scoring only for validation set
if [ $stage -le 4 ]; then
    echo --------------------------
    echo stage 4: score results
    echo --------------------------
    for data_set in $test_set; do
        echo start evaluation for $data_set
        target=$(basename ${data_set%.*})
        result=${expdir}/result_${target}_b${beam}_p${penalty}.json
        reference=${result%.*}_ref.json
        hypothesis=${result%.*}_hyp.json
        result_eval=${result%.*}.eval
        echo Evaluating: $result
        utils/get_annotation.py -s data/stopwords.txt $data_set $reference
        utils/get_hypotheses.py -s data/stopwords.txt $result $hypothesis
        python2 utils/evaluate.py $reference $hypothesis >& $result_eval
        echo Wrote details in $result_eval
        echo "--- summary ---"
        awk '/^(Bleu_[1-4]|METEOR|ROUGE_L|CIDEr):/{print $0; if($1=="CIDEr:"){exit}}'\
            $result_eval
        echo "---------------"
    done
fi
