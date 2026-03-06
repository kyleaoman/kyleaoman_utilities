from PyPDF2 import PdfReader

reader = PdfReader("Downloads/Evelyn_interim_report_KOcomments.pdf")
for npage, page in enumerate(reader.pages, start=1):
    if "/Annots" not in page:
        continue
    print(f"Page {npage:02d}")
    print("-------")
    for _annot in page["/Annots"]:
        annot = _annot.get_object()
        print(f"{annot['/T']}: {annot['/Contents']}")
        print("--")
