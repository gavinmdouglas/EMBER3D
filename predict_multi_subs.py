import torch
import numpy as np
import os
from Bio import SeqIO
import argparse
from Ember3D import Ember3D
from lddt import lddt

parser = argparse.rgumentParser(

    description='''
        Edited script to predict the effects of mutliple substitutions on a protein sequence.
        Note that this requires tables of the substitutions to test, and the positions to test them at.
        There should be one table per protein (in the test-sites folder), encoded as "GENEID.tsv".
        The first field is the substitutions and positions to try per combination, in the format "POS|REF_AA|SUB_AA,POS|REF_AA|SUB_AA...".
        One combination per line. The second table field (split by tabs) is the category of the substituion, which will be output too.

        Please note that this script is not compatible with the original predict_sav.py script, and is intended to be used
        for a different purpose. For that reason the optional outputs are all off by default, and have not been tested
        (and a mutation matrix image cannot be output as multi-mutations are being tested).
        ''',

    formatter_class=argparse.RawDescriptionHelpFormatter

)

parser.add_argument('-i', '--input', dest="input", type=str, required=True)
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('-d', '--device', default='cuda:0', dest="device", type=str)
parser.add_argument('--save-pdb', dest="save_pdb", action="store_true")
parser.add_argument('--save-distance-map', dest="save_distance_map", action="store_true")
parser.add_argument('--save-distance-array', dest="save_distance_array", action="store_true")
parser.add_argument('-m', '--model', default="model/EMBER3D.model", dest='model_checkpoint', type=str)
parser.add_argument('--t5-dir', dest='t5_dir', default="./ProtT5-XL-U50/", type=str)

parser.add_argument('--test-sites', dest='test_sites', default=None, type=str,
                    help='Optional path to folder containing input tables specifying which amino acid positions '
                    'to mutate, and which substitutions to make. '
                    'There should be one table per input gene (named \"GENEID.tsv\"). '
                    'Each table not have a header, and should be tab-delimited with these two columns: '
                    '(1) 1-based AA positions/subs in protein (see format in description), and (2) category of AA sub combo.')

args = parser.parse_args()

# Output directory
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)

Ember3D = Ember3D(args.model_checkpoint, args.t5_dir, args.device)

for i, record in enumerate(SeqIO.parse(args.input, "fasta")):
    id = record.id
    seq = str(record.seq)
    seq_l = list(seq)
    length = len(seq)

    if "X" in seq:
        print("Skipping {} because of unknown residues".format(id))
        continue

    sample_dir = os.path.join(args.output_dir, id)
    if not os.path.isdir(sample_dir):
        os.makedirs(sample_dir)

    pdb_dir = os.path.join(sample_dir, "pdb")
    image_dir = os.path.join(sample_dir, "images")
    dist_dir = os.path.join(sample_dir, "distance_arrays")
    if args.save_pdb and not os.path.isdir(pdb_dir):
        os.makedirs(pdb_dir)
    if args.save_distance_map and not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    if args.save_distance_array and not os.path.isdir(dist_dir):
        os.makedirs(dist_dir)

    # Wild-type prediction
    print("{}\t{}".format(i, id))
    with torch.cuda.amp.autocast():
        result = Ember3D.predict(seq)

        if args.save_pdb:
            result.save_pdb(id, "{}/{}.pdb".format(pdb_dir, id))

        if args.save_distance_map:
            result.save_distance_map("{}/{}.png".format(image_dir, id))

        wild_type_distance_map = result.get_distance_map()

        if args.save_distance_array:
            np.save("{}/{}_distances.npy".format(dist_dir, id), wild_type_distance_map)

    aa_list = list("ACDEFGHIKLMNPQRSTVWY")

    # If test_sites files specified, then restrict analysis to those loci and changes.
    # Read in substitutions to test.
    test_sites_file = args.test_sites + '/' + id + '.tsv'

    # Skip protein if test sites file does not exist.
    if not os.path.isfile(test_sites_file):
        continue

    with open(test_sites_file, 'r') as test_sites_in:
        combos_to_test = test_sites_in.readlines()

    possible_aa = set(aa_list)
    mutation_log = os.path.join(sample_dir, id + "_mutation_log.txt")

    with open(mutation_log, "w") as f:
        for combo in combos_to_test:
            combo_split = combo.strip().split('\t')
            combo_category = combo_split[1]
            subs = combo_split[0].split(',')

            combo_pos = set()

            mut_seq = seq_l.copy()

            for aa_info in subs:

                aa_info_split = aa_info.split('|')
                seq_pos = int(aa_info_split[0])
                seq_idx = seq_pos - 1
                ref_aa = aa_info_split[1]
                test_aa = aa_info_split[2]

                # Make sure there's no input mistake where 'combos' are subs at the same site.
                if seq_pos in combo_pos:
                    raise ValueError("Multiple substitutions at the same position in test_sites file.")
                else:
                    combo_pos.add(seq_pos)

                # Throw error if aa not found in possible set.
                if test_aa not in possible_aa:
                    raise ValueError("Invalid amino acid code {} found in test_sites file.".format(test_aa))

                # Throw error if there is a mismatch between the expected and observed amino acid.
                if ref_aa != seq_l[seq_idx]:
                    raise ValueError("Mismatch between expected and observed amino acid at position {}.".format(seq_pos))

                # Throw error if one of the specified AAs is the actual AA at this position.
                if test_aa == seq_l[seq_idx]:
                    raise ValueError("Specified substitution {} at position {} is the same as the wild-type residue.".format(test_aa, seq_pos))

                mut_seq[seq_idx] = test_aa

            # Create final test sequence after looping through all subs in the combo.
            mut_seq = ''.join(mut_seq)
            mut_id = "{}_{}".format(id, combo.replace('|', '.').replace(',', '-'))

            with torch.cuda.amp.autocast():
                result = Ember3D.predict(mut_seq)

                if args.save_pdb:
                    result.save_pdb(mut_id, "{}/{}.pdb".format(pdb_dir, mut_id))

                if args.save_distance_map:
                    result.save_distance_map("{}/{}.png".format(image_dir, mut_id))

                mutant_distance_map = result.get_distance_map()

                if args.save_distance_array:
                    np.save("{}/{}_distances.npy".format(dist_dir, mut_id), mutant_distance_map)

                struct_diff = lddt(torch.from_numpy(mutant_distance_map), torch.from_numpy(wild_type_distance_map))
                struct_diff = torch.mean(struct_diff).item()
                f.write("{}\t{}\t{}\n".format(combo, combo_category, struct_diff))
