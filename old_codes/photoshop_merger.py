import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from photoshop import Session
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', type=str, help='Directory where output images are present')
    args = parser.parse_args()
    a = os.getcwd() + "\\"
    path = a + args.output_folder

    with Session() as ps:
        # Create new document
        ps.app.preferences.rulerUnits = ps.Units.Pixels
        ps.app.documents.add(1280, 896, name="Stitched")
        ps.app.Load(path + r"\1.exr")
        ps.app.ActiveDocument.Selection.SelectAll()
        ps.app.ActiveDocument.Selection.Copy()
        ps.app.ActiveDocument.Close()
        ps.app.ActiveDocument.Paste()
        ps.app.Load(path + r"\2.exr")
        ps.app.ActiveDocument.Selection.SelectAll()
        ps.app.ActiveDocument.Selection.Copy()
        ps.app.ActiveDocument.Close()
        ps.app.ActiveDocument.Paste()



def open_exr_file(ps, path):
    idOpn = ps.app.charIDToTypeID("Opn ")
    desc386 = ps.ActionDescriptor()
    iddontRecord = ps.app.stringIDToTypeID("dontRecord")
    desc386.putBoolean(iddontRecord, False)
    idforceNotify = ps.app.stringIDToTypeID("forceNotify")
    desc386.putBoolean(idforceNotify, True)
    idnull = ps.app.charIDToTypeID("null")
    desc386.putPath(idnull, path)
    idAs = ps.app.charIDToTypeID("As  ")
    desc387 = ps.ActionDescriptor()
    idioty = ps.app.charIDToTypeID("ioty")
    desc387.putBoolean(idioty, True)
    idiosa = ps.app.charIDToTypeID("iosa")
    desc387.putBoolean(idiosa, False)
    idioac = ps.app.charIDToTypeID("ioac")
    desc387.putBoolean(idioac, False)
    idioal = ps.app.charIDToTypeID("ioal")
    desc387.putBoolean(idioal, False)
    idiocm = ps.app.charIDToTypeID("iocm")
    desc387.putBoolean(idiocm, True)
    idioca = ps.app.charIDToTypeID("ioca")
    desc387.putBoolean(idioca, True)
    idiocd = ps.app.charIDToTypeID("iocd")
    desc387.putBoolean(idiocd, False)
    idioll = ps.app.charIDToTypeID("ioll")
    desc387.putBoolean(idioll, False)
    idioci = ps.app.charIDToTypeID("ioci")
    desc387.putBoolean(idioci, True)
    idiodw = ps.app.charIDToTypeID("iodw")
    desc387.putBoolean(idiodw, False)
    idiocg = ps.app.charIDToTypeID("iocg")
    desc387.putBoolean(idiocg, True)
    idiosr = ps.app.charIDToTypeID("iosr")
    desc387.putBoolean(idiosr, False)
    idioso = ps.app.charIDToTypeID("ioso")
    desc387.putBoolean(idioso, True)
    idiosh = ps.app.charIDToTypeID("iosh")
    desc387.putBoolean(idiosh, True)
    idiocw = ps.app.charIDToTypeID("iocw")
    desc387.putInteger(idiocw, 1000)
    idthreedioExrIO = ps.app.stringIDToTypeID("3d-io Exr-IO")
    desc386.putObject(idAs, idthreedioExrIO, desc387)
    idDocI = ps.app.charIDToTypeID("DocI")
    desc386.putInteger(idDocI, 611)
    ps.app.executeAction(idOpn, desc386)

if __name__ == '__main__':
    main()