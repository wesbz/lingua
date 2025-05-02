python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk29/tt_1.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk29/tt_2.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk29/tt_3.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk29/tt_4.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10

python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk30/val_len_1.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk30/val_len_5.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10

python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk31/val_len_1.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk31/val_len_5.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10

python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk32/cancer.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk32/politician.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk32/sneakers.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk32/social.yaml nodes=2 ngpu=8 mem=0 time=600 partition=learn override=true ncpu=10


# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/135M_val_len_1.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/135M_val_len_5.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/135M_val_len_10.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/360M_val_len_1.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/360M_val_len_5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/360M_val_len_10.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/1.4B_val_len_1.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/1.4B_val_len_5.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37/1.4B_val_len_10.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/135M_val_len_1.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/135M_val_len_5.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/135M_val_len_10.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/360M_val_len_1.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/360M_val_len_5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/360M_val_len_10.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/1.4B_val_len_1.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/1.4B_val_len_5.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk38/1.4B_val_len_10.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/135M_val_len_1.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/135M_val_len_5.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/135M_val_len_10.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/360M_val_len_1.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/360M_val_len_5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/360M_val_len_10.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/1.4B_val_len_1.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/1.4B_val_len_5.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk39/1.4B_val_len_10.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/135M_val_len_1.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/135M_val_len_5.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
# python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/135M_val_len_10.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=false ncpu=10
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/360M_val_len_1.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/360M_val_len_5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/360M_val_len_10.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/1.4B_val_len_1.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/1.4B_val_len_5.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk40/1.4B_val_len_10.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared


python -m lingua.stool script=apps.main.train config=apps/main/configs/smollm-135M.yaml nodes=1 ngpu=8 mem=0 time=1440 override=true ncpu=10 account=ai_society qos=ai_society
python -m lingua.stool script=apps.main.train config=apps/main/configs/smollm-360M.yaml nodes=2 ngpu=8 mem=0 time=1440 override=true ncpu=10 account=ai_society qos=ai_society
python -m lingua.stool script=apps.main.train config=apps/main/configs/smollm-1.4B.yaml nodes=4 ngpu=8 mem=0 time=1440 override=true ncpu=10 account=ai_society qos=ai_society

# Mk41
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/1e-7.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/3e-7.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/1e-6.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/3e-6.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/1e-5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/3e-5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk41/1e-4.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared


# Mk42
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/135M_secret_360M.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/135M_secret_1.4B.yaml nodes=1 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/360M_secret_135M.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/360M_secret_1.4B.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/1.4B_secret_135M.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk42/1.4B_secret_360M.yaml nodes=4 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared


# Mk37-2long
python -m lingua.stool script=apps.main.train config=apps/main/configs/Mk37-2long/360M_val_len_5.yaml nodes=2 ngpu=8 mem=0 time=1440 partition=learn override=true ncpu=10 account=ai_society qos=alignment_shared