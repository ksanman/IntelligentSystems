
#|
=============================================================
module: loader.lisp
description: Loads ANSI Common Lisp Implementation of STRIPS
author: Vladimir Kulyukin
=============================================================
|#

(in-package "USER")

(defparameter *strips-files*
  '("for.lisp"
    "pc.lisp"
    "index.lisp"
    "unify.lisp"
    "database.lisp"
    "store.lisp"
    "show.lisp"
    "binding-structure.lisp"
    "srules.lisp"
    "strips.lisp"
    ;"telescope-operators.lisp"
    ;"telescope-wffs.lisp"
    ))

(defun load-strips ()
  (dolist (file *strips-files*)
    (load file :verbose t :print t)))

;;; end-of-file

