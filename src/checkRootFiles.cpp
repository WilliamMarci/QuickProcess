#include <TFile.h>
#include <TSystemDirectory.h>
#include <TSystemFile.h>
#include <TList.h>
#include <TString.h>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_dir>\n";
        return 1;
    }

    std::ofstream status("status.txt");
    if (!status.is_open()) {
        std::cerr << "Cannot open output file: status\n";
        return 1;
    }

    const char* dirPath = argv[1];
    TSystemDirectory dir("input", dirPath);
    TList* files = dir.GetListOfFiles();
    if (!files) {
        status << "Cannot open directory: " << dirPath << "\n";
        return 1;
    }

    TSystemFile* sysFile = nullptr;
    TIter next(files);
    while ((sysFile = (TSystemFile*)next())) {
        TString name(sysFile->GetName());
        if (sysFile->IsDirectory() || !name.EndsWith(".root")) continue;

        TString fullPath = TString(dirPath) + "/" + name;
        TFile* f = TFile::Open(fullPath, "READ");
        if (f && !f->IsZombie()) {
            status << fullPath.Data() << " : OK\n";
            f->Close();
            delete f;
        } else {
            status << fullPath.Data() << " : FAILED\n";
        }
    }

    delete files;
    status.close();
    return 0;
}

