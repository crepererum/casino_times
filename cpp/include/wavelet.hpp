#pragma once

#include <memory>
#include <string>
#include <vector>

#include <wavelib.h>

#include "utils.hpp"

class wavelet {
    public:
        wavelet(const std::string& wname) {
            _cs_wname = alloc_cs(wname);
            _wave_obj = wave_init(_cs_wname.get());
        }

        ~wavelet() {
            wave_free(_wave_obj);
        }

        wave_object get_ptr() {
            return _wave_obj;
        }

        void print_summary() const {
            wave_summary(_wave_obj);
        }

    private:
        wave_object _wave_obj;
        std::unique_ptr<char[]> _cs_wname;
};

class wavelet_transform {
    public:
        wavelet_transform(const std::shared_ptr<wavelet>& w, const std::string& method, int n, int j) : _w(w) {
            _cs_method = alloc_cs(method);
            _wt_obj = wt_init(_w->get_ptr(), _cs_method.get(), n, j);

            // run dwt once to initialize all structures.
            // not sure if this is an actual bug or undocumented behaviour in wavelib
            std::vector<double> tmp(static_cast<std::size_t>(n), 0.0);
            run_dwt(tmp.data());
        }

        ~wavelet_transform() {
            wt_free(_wt_obj);
        }

        void extension(const std::string& ext) {
            _cs_extname = alloc_cs(ext);
            setDWTExtension(_wt_obj, _cs_extname.get());
        }

        void conv(const std::string& conv) {
            _cs_conv = alloc_cs(conv);
            setWTConv(_wt_obj, _cs_conv.get());
        }

        void run_dwt(const double* inp) {
            // dwt won't change the input, but the API is kinda messy
            dwt(_wt_obj, const_cast<double*>(inp));
        }

        void run_idwt(double* outp) {
            idwt(_wt_obj, outp);
        }

        void print_summary() const {
            wt_summary(_wt_obj);
        }

        std::size_t outlength() const {
            return static_cast<std::size_t>(_wt_obj->outlength);
        }

        const double* output() const {
            return _wt_obj->output;
        }

        double* output() {
            return _wt_obj->output;
        }

    private:
        std::shared_ptr<wavelet> _w;
        wt_object _wt_obj;
        std::unique_ptr<char[]> _cs_method;
        std::unique_ptr<char[]> _cs_extname;
        std::unique_ptr<char[]> _cs_conv;
};
