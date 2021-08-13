;;;test.pdg.el --- Summary

;;; Commentary:
;;; Code:

(pdg
 :variables X Y Z  ; no need to give them values yet.
 :structure X -> Y <- Z
 :cpds
        p(Y|X)
        q(Z|Y)
)



(provide 'test.pdg)
;;; test.pdg.el ends here
