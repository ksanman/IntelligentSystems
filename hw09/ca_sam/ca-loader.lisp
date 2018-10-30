#|
===========================================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-loader.lisp
Author: Vladimir Kulyukin

Bugs to vladimir dot kulyukin at aggiemail dot usu dot edu
===========================================================
|#

(in-package :user)

;;; change this to the directory where the ca files are.
(defparameter *ca-dir* "/home/vladimir/programming/lisp/ca_sam/")

(defparameter *files* '("ca-utilities" "cd" "ca" "ca-functions" "ca-lexicon"))

(defun ca-loader (files &key (mode :lisp))
  (dolist (a-file files t)
    (load (concatenate 'string
            *ca-dir*
            a-file
            (ecase mode
              (:lisp ".lisp")
              (:cl  ".cl")
              (:fasl ".fasl"))))))
