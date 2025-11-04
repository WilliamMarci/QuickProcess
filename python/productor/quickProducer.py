import os
import re
import numpy as np
import math
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection, Object
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module

from ..helpers.utils import deltaPhi, deltaR, deltaR2, deltaEta, closest, polarP4, sumP4, transverseMass, minValue, configLogger, getDigit, closest_pair
from ..helpers.ortHelper import ONNXRuntimeHelper
from ..helpers.nnHelper import convert_prob
from ..helpers.jetmetCorrector import JetMETCorrector, rndSeed
from ..helpers.muonCorrector import MuonScaleResCorrector
from ..helpers.triggerHelper import passTrigger

from .flavTagSFProducer import FlavTagSFProducer
from .leptonSFProducer import TriggerSF, ElectronSFProducer, MuonSFProducer
from .puWeightProducer import PileupWeightProducer
from .eventVetoMapProducer import EventVetoMapProducer
from .topPtWeightProducer import TopPtWeightProducer
from .topSystReweightingProducer import TopSystReweightingProducer
from .renormWeightSFProducer import RenormWeightSFProducer

import logging
logger = logging.getLogger('nano')
configLogger('nano', loglevel=logging.INFO)

lumi_dict = {2015: 19.52, 2016: 16.81, 2017: 41.48, 2018: 59.83}
dataset_dict = {
    'SingleMuon': 100001,
    'SingleElectron': 100002,
    'MuonEG': 100003,
    'DoubleMuon': 100004,
    'DoubleEG': 100005,
    'EGamma': 100006,
    'JetHT': 100007,
    'MET': 100008,
    'BtagCSV': 100009,
}

from .decotaKit import timeit, getentries

class METObject(Object):

    def p4(self):
        return polarP4(self, eta=None, mass=None)


class quickProducer(Module, object):

    def __init__(self, channel='1L', **kwargs):
        self._channel = '1L'
        self._year = 2017
        self.dataset = 'SingleMuon'
        # self.isMC = True
        self._usePuppiJets = True
        self._jmeSysts = {'jec': False, 'jes': None, 'jes_source': '', 'jes_uncertainty_file_prefix': '',
                          'jer': None , 'jmr': None, 'met_unclustered': None, 'applyHEMUnc': False,
                          'smearMET': False}
        self._opts = {
            'min_num_ak4_jets_1L': 4,
            'min_num_ak8_jets_1L': 0,
            'min_num_b_jets': 1,
            'min_num_b_or_c_jets': 3,
            'min_ht_0L': 500,
            'apply_tight_selection': True,
            'apply_score_selection': True, 'apply_qcd_cut': 1e-4,
            'eval_nn': True, 'eval_nn_da_op1': False, 'eval_nn_da_op2': False, 'eval_mlp': False,
            'muon_scale': 'nominal',
            'usePuppiJets': True,
            'fillJetTaggingScores': False, 'fillEventVars': True, 'fillBDTVars': False,
            'runModules': True, 'fillSystWeights': True, 'fillRenormWeights': True, 'fillExtendedPSWeights': True,
        }

        if True:
            self._opts.update({
                'min_num_ak4_jets_1L': 3,
                'min_num_ak8_jets_1L': 1,
                'min_num_b_jets': 1,
                'min_num_b_or_c_jets': 1,
                'apply_tight_selection': False,
                'apply_score_selection': False, 'apply_qcd_cut': None,
                'eval_nn': True, 'eval_nn_da_op1': False, 'eval_nn_da_op2': False, 'eval_mlp': False,
                'muon_scale': 'nominal',
                'fillJetTaggingScores': True, 'fillEventVars': True, 'fillBDTVars': False,
                'runModules': True, 'fillSystWeights': True, 'fillRenormWeights': True, 'fillExtendedPSWeights': True,
            })
        # if self._opts['muon_scale']:
        #     self.muonCorr = MuonScaleResCorrector(year=self._year, corr=self._opts['muon_scale'])

        # ParticleNetAK4 -- exclusive b- and c-tagging categories
        # 5x: b-tagged; 4x: c-tagged; 0: light
        if self._year in (2017, 2018):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.5) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.5) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.5) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.5) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.5) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.5) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.5) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.5) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.2<pn_b_plus_c<=0.5)',
                40: '(0.1<pn_b_plus_c<=0.2)',

                0: '(pn_b_plus_c<=0.1)',
            }
        elif self._year in (2015, 2016):
            self.jetTagWPs = {
                54: '(pn_b_plus_c>0.35) & (pn_b_vs_c>0.99)',
                53: '(pn_b_plus_c>0.35) & (0.96<pn_b_vs_c<=0.99)',
                52: '(pn_b_plus_c>0.35) & (0.88<pn_b_vs_c<=0.96)',
                51: '(pn_b_plus_c>0.35) & (0.70<pn_b_vs_c<=0.88)',
                50: '(pn_b_plus_c>0.35) & (0.40<pn_b_vs_c<=0.70)',

                44: '(pn_b_plus_c>0.35) & (pn_b_vs_c<=0.05)',
                43: '(pn_b_plus_c>0.35) & (0.05<pn_b_vs_c<=0.15)',
                42: '(pn_b_plus_c>0.35) & (0.15<pn_b_vs_c<=0.40)',
                41: '(0.17<pn_b_plus_c<=0.35)',
                40: '(0.1<pn_b_plus_c<=0.17)',

                0: '(pn_b_plus_c<=0.1)',
            }

        if self._usePuppiJets:
            self.puID_WP = None
        else:
            # https://twiki.cern.ch/twiki/bin/view/CMS/PileupJetIDUL
            # NOTE [27.04.2022]: switched back to tight PU ID
            # self.puID_WP = {2015: 1, 2016: 1, 2017: 4, 2018: 4}[self._year]  # L
            # self.puID_WP = {2015: 3, 2016: 3, 2017: 6, 2018: 6}[self._year]  # M
            self.puID_WP = {2015: 7, 2016: 7, 2017: 7, 2018: 7}[self._year]  # T

        # if self._opts['eval_nn']:
        #     prefix = os.path.expandvars('$CMSSW_BASE/src/PhysicsTools/QuickProcess/data')

        #     self.nn_helper = ONNXRuntimeHelper(
        #         preprocess_file='%s/nn/1L/v3/preprocess.json' % prefix,
        #         model_files=['%s/nn/1L/v3/net.%d.onnx' % (prefix, idx) for idx in range(5)])

        # start with modules that should always run, regardless of data or MC
        self._modules = {
            'jetVetomapEventVeto': EventVetoMapProducer,
        }
        # then add those only needed for MC
        if self._opts['runModules']:
            self._modules.update({
                'flavTagSF': FlavTagSFProducer,
                'electronSF': ElectronSFProducer,
                'muonSF': MuonSFProducer,
                'puWeight': PileupWeightProducer,
                'topPtWeight': TopPtWeightProducer,
                'topSystReweighter': TopSystReweightingProducer,
                'renormWeighter': RenormWeightSFProducer,
            })
        for k, cls in self._modules.items():
            self._modules[k] = cls(
                self._year, fillSystWeights=self._opts['fillSystWeights'],
                usePuppiJets=self._opts['usePuppiJets'])
            
        self._needsJMECorr = any([self._jmeSysts['jec'], self._jmeSysts['jes'],
                            self._jmeSysts['jer'], self._jmeSysts['jmr'],
                            self._jmeSysts['met_unclustered'], self._jmeSysts['applyHEMUnc']])

    def evalJetTag(self, j, default=0):
        for wp, expr in self.jetTagWPs.items():
            if eval(expr, j.__dict__):
                return wp
        return default

    def beginJob(self):
    #     if self._needsJMECorr:
    #         self.jetmetCorr.beginJob()
        for mod in self._modules.values():
            mod.beginJob()

    def endJob(self):
        for mod in self._modules.values():
            mod.endJob()
    
    #@timeit
    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.isMC = bool(inputTree.GetBranch('genWeight'))
        self.hasParticleNetAK4 = 'privateNano' if inputTree.GetBranch(
            'Jet_ParticleNetAK4_probb') else 'jmeNano' if inputTree.GetBranch('Jet_particleNetAK4_B') else None
    #     if not self.hasParticleNetAK4 and 'TrigSF' not in self._channel:
    #         raise RuntimeError('No ParticleNetAK4 scores in the input NanoAOD!')
    #     self.rho_branch_name = 'Rho_fixedGridRhoFastjetAll' if bool(
    #         inputTree.GetBranch('Rho_fixedGridRhoFastjetAll')) else 'fixedGridRhoFastjetAll'

    #     self.dataset = None
    #     r = re.search(('mc' if self.isMC else 'data') + r'\/([a-zA-Z0-9_\-]+)\/', inputFile.GetName())
    #     if r:
    #         self.dataset = r.groups()[0]

        self.out = wrappedOutputTree

    #     # NOTE: branch names must start with a lower case letter
    #     # check keep_and_drop_output.txt
    #     self.out.branch("dataset", "I", title=', '.join([f'{k}={v}' for k, v in dataset_dict.items()]))
    #     self.out.branch("year", "I")
    #     self.out.branch("channel", "I")
    #     self.out.branch("lumiwgt", "F")

    #     self.out.branch("passmetfilters", "O")

    #     # triggers for 1L
        self.out.branch("passTrigEl", "O")
        self.out.branch("passTrigMu", "O")
        self.out.branch("passTrigEl1", "O") # [DEBUG] for checking HLT  
        self.out.branch("passTrigEl2", "O") # [DEBUG]

    #     self.out.branch("met", "F")
    #     self.out.branch("met_phi", "F")

    #     # V boson
    #     self.out.branch("v_pt", "F", limitedPrecision=10)
    #     self.out.branch("v_eta", "F", limitedPrecision=10)
    #     self.out.branch("v_phi", "F", limitedPrecision=10)
    #     self.out.branch("v_mass", "F", limitedPrecision=10)

    #     # leptons
    #     self.out.branch("n_lep", "I")

    #     self.out.branch("lep1_pt", "F")
    #     self.out.branch("lep1_eta", "F")
    #     self.out.branch("lep1_phi", "F")
    #     self.out.branch("lep1_mass", "F")
    #     self.out.branch("lep1_etaSC", "F", limitedPrecision=10)
    #     self.out.branch("lep1_pdgId", "I")

    #     # ak4 jets
    #     self.out.branch("n_btag", "I")
    #     self.out.branch("n_ctag", "I")
    #     self.out.branch("n_btagM", "I")
    #     self.out.branch("n_btagT", "I")
    #     self.out.branch("n_ctagM", "I")
    #     self.out.branch("n_ctagT", "I")

    #     self.out.branch("ak4_pt", "F", 20, lenVar="n_ak4")
    #     self.out.branch("ak4_eta", "F", 20, lenVar="n_ak4")
    #     self.out.branch("ak4_phi", "F", 20, lenVar="n_ak4")
    #     self.out.branch("ak4_mass", "F", 20, lenVar="n_ak4")
    #     self.out.branch("ak4_tag", "F", 20, lenVar="n_ak4")
    #     # ak8 jets
    #     self.out.branch("n_ak8", "I")
    #     self.out.branch("ak8_pt", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_eta", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_phi", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_sdmass", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_rawFactor", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_tau21", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_tau32", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_bb", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_cc", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_bc", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_qcd", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_bs", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_cs", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_qq", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_topbw", "F", 10, lenVar="n_ak8")
    #     self.out.branch("ak8_gpt_topw", "F", 10, lenVar="n_ak8")

        for mod in self._modules.values():
            mod.beginFile(inputFile, outputFile, inputTree, wrappedOutputTree)

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        for mod in self._modules.values():
            mod.endFile(inputFile, outputFile, inputTree, wrappedOutputTree)
    #@timeit
    def _correctJetAndMET(self, event):
        if self._needsJMECorr:
            rho = getattr(event, self.rho_branch_name)
            # correct AK4 jets and MET
            self.jetmetCorr.setSeed(rndSeed(event, event._allJets))
            self.jetmetCorr.correctJetAndMET(
                jets=event._allJets,
                lowPtJets=Collection(event, "CorrT1METJet"),
                met=event.met,
                rawMET=METObject(event, "RawPuppiMET") if self._usePuppiJets else METObject(event, "RawMET"),
                defaultMET=METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET"),
                rho=rho, genjets=Collection(event, 'GenJet') if self.isMC else None,
                isMC=self.isMC, runNumber=event.run)
            event._allJets = sorted(event._allJets, key=lambda x: x.pt, reverse=True)  # sort by pt after updating

    #@timeit
    def _selectLeptons(self, event):
        # do lepton selection
        event.looseLeptons = []  # used for jet lepton cleaning & lepton counting

        electrons = Collection(event, "Electron")
        for el in electrons:
            el.etaSC = el.eta + el.deltaEtaSC
            # ttH(bb) analysis uses tight electron ID
            # if el.pt > 15 and abs(el.eta) < 2.4 and el.cutBased == 4:
            # NOTE: try mvaFall17V2Iso_WP90
            if 1.4442 <= abs(el.etaSC) <= 1.5560:
                continue
            if el.pt > 15 and abs(el.eta) < 2.4 and el.mvaFall17V2Iso_WP90:
                el._wp_ID = 'wp90iso'
                event.looseLeptons.append(el)

        muons = Collection(event, "Muon")
        for mu in muons:
            # if self._opts['muon_scale']:
            #     self.muonCorr.correct(event, mu, self.isMC)
            if mu.pt > 15 and abs(mu.eta) < 2.4 and mu.tightId and mu.pfRelIso04_all < 0.25:
                mu._wp_ID = 'TightID'
                mu._wp_Iso = 'LooseRelIso'
                event.looseLeptons.append(mu)

        event.looseLeptons.sort(key=lambda x: x.pt, reverse=True)
    #@timeit
    def _preSelect(self, event):
        event.selectedLeptons = []  # used for reconstructing the top quarks
        NUM_LEP = 1
        if len(event.looseLeptons) != NUM_LEP:
            return False
        
        electrons = []
        muons = []
        for lep in event.looseLeptons:
            if abs(lep.pdgId) == 13:
                # mu (26/29/26 GeV)
                muPtCut = 29 if self._year == 2017 else 26
                if lep.pt > muPtCut and lep.tightId and lep.pfRelIso04_all < 0.15:
                    lep._wp_Iso = 'TightRelIso'
                    event.selectedLeptons.append(lep)
                    muons.append(lep)
            else:
                # ele (29/30/30 GeV)
                ePtCut = 30 if self._year in (2017, 2018) else 29
                # ttH(bb) analysis uses tight electron ID
                # if lep.pt > ePtCut and lep.cutBased == 4:
                # NOTE: try mvaFall17V2Iso_WP80
                if lep.pt > ePtCut and lep.mvaFall17V2Iso_WP80:
                    lep._wp_ID = 'wp80iso'
                    event.selectedLeptons.append(lep)
                    electrons.append(lep)
        if len(event.selectedLeptons) != NUM_LEP:
            return False
        return True
    
    #@timeit
    def _cleanObjects(self, event):
        event.ak4jets = []
        for j in event._allJets:
            if not (j.pt > 25 and abs(j.eta) < 2.4 and (j.jetId & 4)):
                # NOTE: ttH(bb) uses jets w/ pT > 30 GeV, loose PU Id
                # pt, eta, tightIdLepVeto, loose PU ID
                continue
            if not self._usePuppiJets and not (j.pt > 50 or j.puId >= self.puID_WP):
                # apply jet puId only for CHS jets
                continue
            if closest(j, event.looseLeptons)[1] < 0.4:
                continue
            j.btagDeepFlavC = j.btagDeepFlavB * j.btagDeepFlavCvB / (
                1 - j.btagDeepFlavCvB) if (j.btagDeepFlavCvB >= 0 and j.btagDeepFlavCvB < 1) else -1
            if self.hasParticleNetAK4 == 'privateNano':
                # attach ParticleNet scores
                j.pn_b = convert_prob(j, ['b', 'bb'], ['c', 'cc', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_c = convert_prob(j, ['c', 'cc'], ['b', 'bb', 'uds', 'g'], 'ParticleNetAK4_prob')
                j.pn_uds = convert_prob(j, 'uds', ['b', 'bb', 'c', 'cc', 'g'], 'ParticleNetAK4_prob')
                j.pn_g = convert_prob(j, 'g', ['b', 'bb', 'c', 'cc', 'uds'], 'ParticleNetAK4_prob')
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            elif self.hasParticleNetAK4 == 'jmeNano':
                # attach ParticleNet scores
                j.pn_b = j.particleNetAK4_B
                j.pn_c = j.particleNetAK4_B * j.particleNetAK4_CvsB / (
                    1 - j.particleNetAK4_CvsB) if (j.particleNetAK4_CvsB >= 0 and j.particleNetAK4_CvsB < 1) else -1
                j.pn_uds = np.clip(1 - j.pn_b - j.pn_c, 0, 1) * j.particleNetAK4_QvsG if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_g = np.clip(1 - j.pn_b - j.pn_c - j.pn_uds, 0, 1) if (
                    j.particleNetAK4_QvsG >= 0 and j.particleNetAK4_QvsG < 1) else -1
                j.pn_b_plus_c = j.pn_b + j.pn_c
                j.pn_b_vs_c = j.pn_b / j.pn_b_plus_c
                j.tag = self.evalJetTag(j)
            else:
                j.tag = 0
            event.ak4jets.append(j)

        event.ak4_b_or_c_jets = []
        event.ak4_b_jets = []
        event.ak4_c_jets = []

        for jet_idx, j in enumerate(event.ak4jets):
            j.idx = jet_idx
            if j.tag > 0:
                event.ak4_b_or_c_jets.append(j)
                if j.tag >= 50:
                    event.ak4_b_jets.append(j)
                if 40 <= j.tag < 50:
                    event.ak4_c_jets.append(j)

        # ak8 jets
        event.ak8jets = []
        # print("Number of fat jets before selection: ", len(event._allFatJets))
        for j in event._allFatJets:
            # print(f"Fat jet pt: {j.pt}, eta: {j.eta}, jetId: {j.jetId}")
            if not (j.pt > 200 and abs(j.eta) < 2.4 and (j.jetId & 2)):  #[DIFF] tt is 2
                continue
            if closest(j, event.looseLeptons)[1] < 0.8:
                continue
            # attach GloParT scores
            j.gpt_bb = j.inclParTMDV2_probHbb
            j.gpt_cc = j.inclParTMDV2_probHcc
            j.gpt_bc = j.inclParTMDV2_probHbc
            j.gpt_qcd = convert_prob(j, None, ['QCDbb', 'QCDb', 'QCDcc', 'QCDc', 'QCDothers'], 'inclParTMDV2_prob')
            j.gpt_topbw = convert_prob(j, None, ['TopbWcs', 'TopbWqq', 'TopbWq', 'TopbWs', 'TopbWc', 'TopbWev', 'TopbWmv', 'TopbWtauev', 'TopbWtauhv', 'TopbWtaumv'], 'inclParTMDV2_prob')
            j.gpt_topw = convert_prob(j, None, ['TopWcs', 'TopWqq', 'TopWev', 'TopWmv', 'TopWtauev', 'TopWtauhv', 'TopWtaumv'], 'inclParTMDV2_prob')
            j.gpt_bs = j.inclParTMDV2_probHbs
            j.gpt_cs = j.inclParTMDV2_probHcs
            j.gpt_qq = j.inclParTMDV2_probHqq
            event.ak8jets.append(j)
    #@timeit
    def _selectEvent(self, event):
        # logger.debug('processing event %d' % event.event)
        event.Vboson = None

        event.Vboson = polarP4(event.selectedLeptons[0]) + (event.met.p4())

        if len(event.ak4jets) < self._opts['min_num_ak4_jets_1L']:
            return False
        if len(event.ak8jets) < self._opts['min_num_ak8_jets_1L']:
            return False
        if len(event.ak4_b_jets) < self._opts['min_num_b_jets']:
            return False
        if len(event.ak4_b_or_c_jets) < self._opts['min_num_b_or_c_jets']:
            return False
        if event.met.pt < 20:
            return False

        return True
    
    #@timeit
    def _selectTriggers(self, event):
        out_data = {}
        if self._year == 2017:
            flagL1DoubleEG = False
            for obj in Collection(event, "TrigObj"):
                if (obj.id == 11) and (obj.filterBits & 1024):
                    # 1024 = 1e (32_L1DoubleEG_AND_L1SingleEGOr)
                    flagL1DoubleEG = True
                    break
            event.HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag = event.HLT_Ele32_WPTight_Gsf_L1DoubleEG and flagL1DoubleEG

            out_data["passTrigEl"] = passTrigger(
                event, ['HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag'])#, 'HLT_Ele28_eta2p1_WPTight_Gsf_HT150'])
            out_data["passTrigEl1"] = passTrigger(event, 'HLT_Ele32_WPTight_Gsf_L1DoubleEG_L1Flag')     #[DEBUG]
            out_data["passTrigEl2"] = passTrigger(event, 'HLT_Ele28_eta2p1_WPTight_Gsf_HT150')          #[DEBUG]
            out_data["passTrigMu"] = passTrigger(event, 'HLT_IsoMu27')

        # apply trigger selections on data
        if not self.isMC and self.dataset is not None:
            if self._channel == '1L':
                passTrig1L = False
                if self.dataset in ('EGamma', 'SingleElectron'):
                    passTrig1L = out_data['passTrigEl']
                elif self.dataset == 'SingleMuon':
                    passTrig1L = out_data['passTrigMu']
                if not passTrig1L:
                    return False

        for key in out_data:
            self.out.fillBranch(key, out_data[key])
        return True

    # def _fillEventInfo(self, event):
    #     out_data = {}

    #     out_data["dataset"] = dataset_dict[self.dataset] if self.dataset in dataset_dict else 1 if self.dataset else 0
    #     out_data["year"] = self._year
    #     out_data["channel"] = int(self._channel[0])
    #     out_data["lumiwgt"] = lumi_dict[self._year]

    #     # met filters -- updated for UL
    #     met_filters = bool(
    #         event.Flag_goodVertices and
    #         event.Flag_globalSuperTightHalo2016Filter and
    #         event.Flag_HBHENoiseFilter and
    #         event.Flag_HBHENoiseIsoFilter and
    #         event.Flag_EcalDeadCellTriggerPrimitiveFilter and
    #         event.Flag_BadPFMuonFilter and
    #         event.Flag_BadPFMuonDzFilter and
    #         event.Flag_eeBadScFilter
    #     )
    #     if self._year in (2017, 2018):
    #         met_filters = met_filters and event.Flag_ecalBadCalibFilter
    #     out_data["passmetfilters"] = met_filters

    #     if self.isMC:
    #         # L1 prefire weights
    #         out_data["l1PreFiringWeight"] = event.L1PreFiringWeight_Nom
    #         if self._opts['fillSystWeights']:
    #             out_data["l1PreFiringWeightUp"] = event.L1PreFiringWeight_Up
    #             out_data["l1PreFiringWeightDown"] = event.L1PreFiringWeight_Dn

    #         # trigger SFs
    #         if self._channel in ('0L', '1L', '2L'):
    #             trigWgt = self.trigSF.get_trigger_sf(event)
    #             out_data["trigEffWeight"] = trigWgt[0]
    #             if self._opts['fillSystWeights']:
    #                 out_data["trigEffWeightUp"] = trigWgt[1]
    #                 out_data["trigEffWeightDown"] = trigWgt[2]

    #     # met
    #     out_data["met"] = event.met.pt
    #     out_data["met_phi"] = event.met.phi

    #     # V boson
    #     out_data["v_pt"] = event.Vboson.Pt()
    #     out_data["v_eta"] = event.Vboson.Eta()
    #     out_data["v_phi"] = event.Vboson.Phi()
    #     out_data["v_mass"] = event.Vboson.M()

    #     # leptons
    #     out_data["n_lep"] = len(event.looseLeptons)
    #     if len(event.selectedLeptons) > 0:
    #         out_data["lep1_pt"] = event.selectedLeptons[0].pt
    #         out_data["lep1_eta"] = event.selectedLeptons[0].eta
    #         out_data["lep1_etaSC"] = event.selectedLeptons[0].etaSC if abs(
    #             event.selectedLeptons[0].pdgId) == 11 else -999
    #         out_data["lep1_phi"] = event.selectedLeptons[0].phi
    #         out_data["lep1_mass"] = event.selectedLeptons[0].mass
    #         out_data["lep1_pdgId"] = event.selectedLeptons[0].pdgId


    #     # AK4 jets, cleaned vs leptons
    #     out_data["n_btag"] = len(event.ak4_b_jets)
    #     out_data["n_ctag"] = len(event.ak4_c_jets)

    #     out_data["n_btagM"] = sum(j.tag >= 51 for j in event.ak4_b_jets)
    #     out_data["n_btagT"] = sum(j.tag >= 52 for j in event.ak4_b_jets)

    #     out_data["n_ctagM"] = sum(j.tag >= 41 for j in event.ak4_c_jets)
    #     out_data["n_ctagT"] = sum(j.tag >= 42 for j in event.ak4_c_jets)

    #     ak4_pt = []
    #     ak4_eta = []
    #     ak4_phi = []
    #     ak4_mass = []
    #     ak4_tag = []
    #     ak4_hflav = []
    #     ak4_bdisc = []
    #     ak4_cvbdisc = []
    #     ak4_cvldisc = []
    #     ak4_pn_b = []
    #     ak4_pn_c = []
    #     ak4_pn_uds = []
    #     ak4_pn_g = []

    #     for j in event.ak4jets:
    #         ak4_pt.append(j.pt)
    #         ak4_eta.append(j.eta)
    #         ak4_phi.append(j.phi)
    #         ak4_mass.append(j.mass)
    #         ak4_tag.append(j.tag)
    #         if self.isMC:
    #             ak4_hflav.append(j.hadronFlavour)

    #         if self._opts['fillJetTaggingScores']:
    #             ak4_bdisc.append(j.btagDeepFlavB)
    #             ak4_cvbdisc.append(j.btagDeepFlavCvB)
    #             ak4_cvldisc.append(j.btagDeepFlavCvL)
    #             if self.hasParticleNetAK4:
    #                 ak4_pn_b.append(j.pn_b)
    #                 ak4_pn_c.append(j.pn_c)
    #                 ak4_pn_uds.append(j.pn_uds)
    #                 ak4_pn_g.append(j.pn_g)

    #     out_data["ak4_pt"] = ak4_pt
    #     out_data["ak4_eta"] = ak4_eta
    #     out_data["ak4_phi"] = ak4_phi
    #     out_data["ak4_mass"] = ak4_mass
    #     out_data["ak4_tag"] = ak4_tag


    #     # ak8 jets
    #     ak8_pt = []
    #     ak8_eta = []
    #     ak8_phi = []
    #     ak8_sdmass = []
    #     ak8_rawFactor = []
    #     ak8_tau21 = []
    #     ak8_tau32 = []
    #     ak8_gpt_bb = []
    #     ak8_gpt_cc = []
    #     ak8_gpt_bc = []
    #     ak8_gpt_qcd = []
    #     ak8_gpt_topbw = []
    #     ak8_gpt_topw = []
    #     ak8_gpt_bs = []
    #     ak8_gpt_cs = []
    #     ak8_gpt_qq = []

    #     for j in event.ak8jets:
    #         ak8_pt.append(j.pt)
    #         ak8_eta.append(j.eta)
    #         ak8_phi.append(j.phi)
    #         ak8_sdmass.append(j.msoftdrop)
    #         ak8_rawFactor.append(j.rawFactor)
    #         ak8_tau21.append(j.tau2 / j.tau1 if j.tau1 > 0 else 99)
    #         ak8_tau32.append(j.tau3 / j.tau2 if j.tau2 > 0 else 99)
    #         ak8_gpt_bb.append(j.gpt_bb)
    #         ak8_gpt_cc.append(j.gpt_cc)
    #         ak8_gpt_bc.append(j.gpt_bc)
    #         ak8_gpt_qcd.append(j.gpt_qcd)
    #         ak8_gpt_topbw.append(j.gpt_topbw)
    #         ak8_gpt_topw.append(j.gpt_topw)
    #         ak8_gpt_bs.append(j.gpt_bs)
    #         ak8_gpt_cs.append(j.gpt_cs)
    #         ak8_gpt_qq.append(j.gpt_qq)

    #     # out_data["n_allak8"] = len(event._allFatJets)
    #     out_data["n_ak8"] = len(event.ak8jets)
    #     out_data["ak8_pt"] = ak8_pt
    #     out_data["ak8_eta"] = ak8_eta
    #     out_data["ak8_phi"] = ak8_phi
    #     out_data["ak8_sdmass"] = ak8_sdmass
    #     out_data["ak8_rawFactor"] = ak8_rawFactor
    #     out_data["ak8_tau21"] = ak8_tau21
    #     out_data["ak8_tau32"] = ak8_tau32
    #     out_data["ak8_gpt_bb"] = ak8_gpt_bb
    #     out_data["ak8_gpt_cc"] = ak8_gpt_cc
    #     out_data["ak8_gpt_bc"] = ak8_gpt_bc
    #     out_data["ak8_gpt_qcd"] = ak8_gpt_qcd
    #     out_data["ak8_gpt_topbw"] = ak8_gpt_topbw
    #     out_data["ak8_gpt_topw"] = ak8_gpt_topw
    #     out_data["ak8_gpt_bs"] = ak8_gpt_bs
    #     out_data["ak8_gpt_cs"] = ak8_gpt_cs
    #     out_data["ak8_gpt_qq"] = ak8_gpt_qq


    #     for key in out_data:
    #         self.out.fillBranch(key, out_data[key])

    #@timeit
    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        event.idx = event._entry if event._tree._entrylist is None else event._tree._entrylist.GetEntry(event._entry)
        event._allJets = Collection(event, "Jet")
        event._allFatJets = Collection(event, "FatJet")
        event.met = METObject(event, "PuppiMET") if self._usePuppiJets else METObject(event, "MET")

        self._selectLeptons(event)
        if self._preSelect(event) is False:
            return False
        if self._selectTriggers(event) is False:
            return False

        self._correctJetAndMET(event)
        self._cleanObjects(event)
        if self._selectEvent(event) is False:
            return False

        # for mod in self._modules.values():
        #     assert mod.analyze(event)

        return True


def quickFromConfig():
    return quickProducer()
