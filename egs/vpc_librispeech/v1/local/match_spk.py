
spkid_file = "data/spkid.txt"
valid_egs_file = "exp/xvector_nnet_1a_kadv5_rm457/egs/valid_diagnostic.scp"
valid_egs_file_adv = "exp/xvector_nnet_1a_kadv5_rm457/egs/valid_diagnostic_adv.scp"

with open(spkid_file) as f1, open(valid_egs_file) as f2, open(valid_egs_file_adv, 'w') as f3:
    lines1 = f1.read().splitlines()
    lines2 = f2.read().splitlines()

    spk251 = set([x.split()[0] for x in lines1])
    spk_valid = set([x.split()[0].split('-')[0] for x in lines2])

    print(spk251 & spk_valid)

    for l in lines2:
        if l.split()[0].split('-')[0] in spk251:
            f3.write(l + '\n')


