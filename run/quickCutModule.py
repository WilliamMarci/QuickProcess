#!/usr/bin/env python3
import subprocess
import os
import json

def _base_cut(choose='1L'):
    cut_dict = {
        'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        # 'ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.5 && Electron_mvaFall17V2Iso_WP90'
                #    ' && (abs(Electron_eta+Electron_deltaEtaSC)<1.4442 || abs(Electron_eta+Electron_deltaEtaSC)>1.5560)',
        'mu_cut': 'Muon_pt>10 && abs(Muon_eta)<2.4 && Muon_tightId && Muon_pfRelIso04_all<0.25',
        # 'mu_cut': 'Muon_pt>10 && abs(Muon_eta)<2.4', #&& Muon_tightId && Muon_pfRelIso04_all<0.25',
        'tight_ele_cut': 'Electron_pt>25 && Electron_mvaFall17V2Iso_WP80',
        'tight_mu_cut': 'Muon_pt>20 && Muon_pfRelIso04_all<0.15',
        'loose_ele_cut': 'Electron_pt>15 && abs(Electron_eta)<2.4 && Electron_mvaFall17V2Iso_WP90',
        'loose_mu_cut': 'Muon_pt>15 && Muon_pfRelIso04_all<0.25 && Muon_tightId',
        'jet_count': 'Sum$(Jet_pt>15 && abs(Jet_eta)<2.4 && (Jet_jetId & 4))',
        'fatjet_count': 'Sum$(FatJet_pt>150 && abs(FatJet_eta)<2.4)' ,# && (FatJet_jetId & 2) && FatJet_msoftdrop>30)',
    }
    basesels ={
        '1L': '(Sum$({ele_cut} && {tight_ele_cut}) + Sum$({mu_cut} && {tight_mu_cut})) >= 1 && '
              '{jet_count} >= 3 && {fatjet_count} >= 1',
        '1L-Tight': '(Sum$({loose_ele_cut})+ Sum$({loose_mu_cut})) == 1 && '
              '(Sum$({ele_cut} && {tight_ele_cut}) + Sum$({mu_cut} && {tight_mu_cut})) >= 1 && '
              '{jet_count} >= 3 && {fatjet_count} >= 1',
    }
    cut = basesels[choose].format(**cut_dict)
    return cut
golden_json = {
    '2015': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2016': 'Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON.txt',
    '2017': 'Cert_294927-306462_13TeV_UL2017_Collisions17_GoldenJSON.txt',
    '2018': 'Cert_314472-325175_13TeV_Legacy2018_Collisions18_JSON.txt',
}

def main(args):
    with open(args.metadata, 'r') as fp:
        try:
            md = json.load(fp)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Failed to load quickCut.json: {e}")
            return
    modules = []
    for mod, names in md['imports']:
        print(mod, names)
        modules.append(f'-I {mod} {names}')

    input_path = args.inputdir
    output_dir = args.output
    year = args.year
    output_data_dir = os.path.join(output_dir, year, 'Data')
    output_mc_dir = os.path.join(output_dir, year, 'MC')

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            os.makedirs(output_data_dir)
            os.makedirs(output_mc_dir)
            os.makedirs(os.path.join(output_data_dir,"result"))
            os.makedirs(os.path.join(output_mc_dir,"result"))
        except Exception as e:
            print(f"[ERROR] Could not create output directories: {e}")
            return

    input_files = []
    if os.path.isdir(input_path):
        for f in os.listdir(input_path):
            if f.endswith('.root'):
                input_files.append(os.path.join(input_path, f))
    elif os.path.isfile(input_path):
        input_files.append(input_path)
    else:
        raise ValueError("Input path is neither a file nor a directory")

    print(f"[INFO] Found {len(input_files)} input files.")


    if args.random_choice > 0:
        import random
        random.seed(42)
        input_files = random.sample(input_files, min(args.random_choice, len(input_files)))
        print(f"[INFO] Randomly selected {len(input_files)} input files.")

    cut = md['cut']
    json_path = '../data/JSON/'
    json_file = golden_json[year] if args.type == 'Data' else None
    json_input = os.path.join(json_path, json_file) if json_file else None
    print(f"[INFO] Using cut: {cut}")

    for input_file in input_files:
        cmd = "nano_postproc.py -c '{cut}' --bi {branchsel} --bo {outputbranchsel} -z {compression} {jsonInput} --first-entry {firstEntry} {outputDir} {inputFiles} {modules}".format(
            outputDir=output_data_dir if args.type == 'Data' else output_mc_dir,
            inputFiles=input_file,
            cut=cut,
            branchsel=os.path.basename(md['branchsel_in']),
            outputbranchsel=os.path.basename(md['branchsel_out']),
            compression=md.get('compression', 'LZMA:9'),
            jsonInput='-J %s' % json_input if json_input else '',
            firstEntry=md.get('firstEntry', 0),
            modules=' '.join(modules),
        )
        print(f"[INFO] Processing file: {input_file}")
        p = subprocess.Popen(cmd, shell=True)
        p.communicate()
        if p.returncode != 0:
            print(f"[ERROR] Failed to process file {input_file}!")
            continue
    # hadd files
    final_output = os.path.join(output_data_dir if args.type == 'Data' else output_mc_dir,"result" ,f"processed_{year}_{args.type}_{args.choose_cut}.root")
    hadd_cmd = f"haddnano.py  {final_output} {os.path.join(output_data_dir if args.type == 'Data' else output_mc_dir, '*.root')}"
    print(f"[INFO] Merging files into {final_output}")
    try:        
        subprocess.run(hadd_cmd, shell=True, check=True)
        print(f"[INFO] Merging completed successfully.")
        # remove all Skim files
        for f in os.listdir(output_data_dir if args.type == 'Data' else output_mc_dir):
            if f.endswith('Skim.root'):
                os.remove(os.path.join(output_data_dir if args.type == 'Data' else output_mc_dir, f))
                print(f"[INFO] Removed intermediate file: {f}")
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Merging failed: {e}")
    print(f"[INFO] Processing completed. Output files are in {output_dir}")

def create_metadata(args):
        modules = [[
            "PhysicsTools.QuickProcess.productor.quickProducer",
            "quickFromConfig"
        ],]
        metadata = {
            'cut' : _base_cut(args.choose_cut),
            'branchsel_in': 'keep_and_drop_input.txt',
            'branchsel_out': 'keep_and_drop_output.txt',
            'compression': 'LZMA:9',
            'year': args.year,
            'type': args.type,
            'imports': modules
        }
        # save metadata to JSON file
        metadata_path = args.metadata
        try:
            with open(metadata_path, 'w') as fp:
                json.dump(metadata, fp, indent=4)
            print(f"[INFO] Metadata saved to {metadata_path}")        
        except Exception as e:
            print(f"[ERROR] Could not save metadata: {e}")
    


def get_arg_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Run quickCut on nanoAOD files')
    parser.add_argument('--inputdir', '-i', required=True, help='Input nanoAOD file or directory')
    parser.add_argument('--output', '-o', required=True, help='Output directory')
    parser.add_argument('--year', '-y', required=True, choices=['2015', '2016', '2017', '2018'], help='Data taking year')
    parser.add_argument('--random-choice', '-r', type=int, default=-1, help='Randomly select N files from input')   
    parser.add_argument('--type', '-t', choices=['Data', 'MC'], required=True, help='Type of input files')
    parser.add_argument('--create-metadata', '--create', action='store_true', help='Create metadata JSON file for processed files')
    parser.add_argument('--metadata', '-m', required=True, help='Path to metadata JSON file')
    parser.add_argument('--choose-cut', '-c', choices=['1L', '1L-Tight'], default='1L', help='Choose cut configuration')
    args= parser.parse_args()
    return args

if __name__ == '__main__':
    args = get_arg_parser()
    if args.create_metadata:
        create_metadata(args)
    else:
        main(args)