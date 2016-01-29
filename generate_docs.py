import pdoc
import os

DOCS_ROOT = 'docs'

TARGET_MODULES = ['von_mises_fisher']

def generate_doc(module_name):
    rd_doc = pdoc.html('von_mises_fisher.{0}'.format(module_name))
    rd_doc_filename = os.path.join(DOCS_ROOT, module_name) + '.html'
    f = open(rd_doc_filename, 'w')
    f.write(rd_doc)
    f.close()

if __name__ == '__main__':
    for m in TARGET_MODULES:
        generate_doc(m)
