#|
==============================================
-*- Mode: Lisp; Syntax: Common-Lisp -*-

File: ca-defs.lisp
Author: Vladimir Kulyukin
Description: Simple word definitions for CA
===============================================
|#

(in-package :user)

(define-ca-word 
    jack
    (concept nil (human :name (jack) :sex (male))))

(define-ca-word
    john
    (concept nil (human :name (john) :sex (male))))

(define-ca-word
    ann
    ;;; your code here
    )

(define-ca-word 
    ate
  (concept ?act (ingest :time (past)))
  (request (test (before ?act ?actor (animate)))
           (actions (modify ?act :actor ?actor)))
  (request (test (after ?act ?food (food)))
	   (actions (modify ?act :object ?food))))

(define-ca-word
    bought
    ;;; your code here
    )

(define-ca-word
    kite
    ;;; your code here
    )

(define-ca-word
    store
    ;;; your code here
    )

(define-ca-word
    went
    (concept ?act (ptrans :time (past)))
    (request (test (before ?act ?actor (animate)))
             (actions (modify ?act :actor ?actor)))
    (request (test (before ?act ?actor (animate)))
	     (actions (modify ?act :object ?actor)))
    (request (test (after ?act ?dir (direction)))
             (actions (modify ?act :to ?dir)))
    (request (test (after ?act ?loc (location)))
	     (actions (modify ?act :to ?loc))))

(define-ca-word
    restaurant
    (concept nil (restaurant)))

(define-ca-word
    home
    (concept nil (home)))

(define-ca-word
    to
    (concept ?to (to))
    (request (test (and (after ?to ?loc (location))
			(before ?to ?ptrans (ptrans))))
             (actions (modify ?ptrans :to ?loc))))

(define-ca-word 
    apple
    (concept nil (apple)))

(define-ca-word
    pear
    (concept nil (pear)))

(define-ca-word 
    an
    (mark ?x)
    (request (test (after ?x ?con (concept)))
	     (actions (modify ?con :ref (indef))))
    (request (test (after ?x ?con (concept)))
	     (actions (modify ?con :number (singular)))))

(define-ca-word
    a
    (mark ?x)
    (request (test (after ?x ?loc (location)))
             (actions (modify ?loc :ref (indef))))
    (request (test (after ?x ?loc (location)))
             (actions (modify ?loc :number (singular))))
    (request (test (after ?x ?obj (phys-obj)))
	     (actions (modify ?obj :ref (indef))))
    (request (test (after ?x ?obj (phys-obj)))
	     (actions (modify ?obj :number (singular)))))

(define-ca-word
    lobster
    (concept nil (lobster)))

;;; end-of-file

