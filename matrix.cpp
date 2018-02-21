#include "matrix.h"
#include <cmath>

matrix::matrix(): row_num(0), col_num(0){};


matrix::matrix(int r, int c): row_num(r), col_num(c), data(std::vector<double>(r*c)){};


matrix matrix::zeros(int n1, int n2){
    matrix m(n1,n2);

    for (int k = 0; k < m.size(); k++){
        m.data[k] = 0.0;
    }
    return m;
}

matrix matrix::ones(int n1, int n2){
    matrix m(n1,n2);

    for (int k = 0; k < m.size(); k++){
        m.data[k] = 1.0;
    }
    return m;
}


matrix matrix::random(int n1, int n2){
    matrix m(n1,n2);

    for (int k = 0; k < m.size(); k++){
        m.data[k] = (std::rand() % 50)/25.0 - 1.0;
    }
    return m;
}


matrix matrix::identity(int n){
    matrix m = matrix::zeros(n,n);

    for (int k = 0; k < m.rows(); k++){
        m(k,k) = 1.0;
    }
    return m;
}


int matrix::rows() const {
    return row_num;
}


int matrix::cols() const {
    return col_num;
}


int matrix::size() const {
    return data.size();
}


double& matrix::operator()(int n){
    return data.at(n);
}



double matrix::operator()(int n) const {
    return data[n];
}



double& matrix::operator()(int r, int c){
    return data.at(r*col_num + c);
}



double matrix::operator()(int r, int c) const {
    return data[r*col_num + c];
}


void matrix::add_row(int n, const matrix &m){

    if (col_num != m.cols() ||  m.rows()>1){
        throw Exception("matrix error with add_row: col_num != m.cols() ||  m.rows()>1");
    };

	for (int k=0; k < col_num; k++){
		auto it = data.begin() + n*col_num + k;
        data.insert(it,m(1,k));
	}

    row_num++;
};




void matrix::add_col(int n, const matrix &m ){

    if (row_num != m.rows() ||  m.cols()>1){
        throw Exception("matrix error with add_col: row_num != m.rows() ||  m.cols()>1");
    };

	for (int k=0; k < row_num; k++){
		auto it = data.begin() + k*col_num + n;
        data.insert(it,m(k,1));
        n++;
	}

    col_num++;
}



void matrix::h_concatenate(const matrix & m){

    if (row_num != m.rows()){
        throw Exception("matrix error with horizontal concatenate: row_num != m.rows()");
    }


    for (int k=0; k < row_num; k++){
        int n = (k+1)*col_num + k*m.cols();
        for (int l=0; l < m.cols(); l++){
            auto it = data.begin() + n + l;
            data.insert(it,m(k,l));
        }
	}

    col_num += m.cols();
}



void matrix::v_concatenate(const matrix & m){

    if (col_num != m.cols()){
        throw Exception("matrix error with vertical concatenate: col_num != m.cols()");
    }

    for (int k=0; k < row_num; k++){
        for (int l=0; l < m.cols(); l++){
            auto it = data.end();
            data.insert(it,m(k,l));
        }
    }
    row_num += m.rows();
}



void matrix::remove_row(int n){

    auto it = data.begin() + n*col_num;
    data.erase(it,it + col_num);

    row_num--;
}



void matrix::remove_col(int n){

    auto it = data.begin() + n;

    for (int k=0; k < row_num; k++){
        it = data.erase(it) + col_num - 1;
    }

    col_num--;
}


matrix& matrix::operator+=(const matrix& m){

    if (row_num != m.rows() || col_num!= m.cols()){
        throw Exception("matrix error: dimension mismatch with +=");
    }

    for (int k=0; k < data.size(); k++){
        data[k] += m.data[k];
    }

    return *this;
}


matrix& matrix::operator*=(const double& t){

    for (int k=0; k < data.size(); k++){
        data[k] *= t;
    };

    return *this;
}


matrix& matrix::operator/=(const double& t){

    for (int k=0; k < data.size(); k++){
        data[k] /= t;
    };

    return *this;
}


std::vector<double> matrix::return_data() const {
    return data;
}


matrix matrix::transpose() const {
    matrix _m(col_num,row_num);

    for (int k = 0; k < col_num; k++){
        for (int l = 0; l < row_num; l++){
            _m(k,l) = (*this)(l,k);
        }
    }
    return _m;
}


void matrix::block(int row_start, int col_start, const matrix& m){

    if (row_num < row_start + m.rows() || col_num < col_start + m.cols()){
        throw Exception("matrix error: input matrix too large in block function");
    }

    for (int x = 0; x < m.rows(); x++){
        int r = row_start + x;
        for (int y = 0; y < m.cols(); y++){
            int c = col_start + y;
            data[r*col_num + c] = m(x,y);
        }
    }
}




matrix matrix::block(int row_start, int col_start, int row_length, int col_length){

    matrix m(row_length,col_length);

    for (int x = 0; x < row_length; x++){
        int r = row_start + x;
        for (int y = 0; y < col_length; y++){
            int c = col_start + y;
            m(x,y)= data[r*col_num + c];
        }
    }
    return m;
}


matrix matrix::row(int n){

    matrix m(1,col_num);

    for (int x = 0; x < col_num; x++){
        m(0,x)= data[n*col_num + x];
    }

    return m;
}


matrix matrix::col(int n){

    matrix m(row_num,1);

    for (int x = 0; x < row_num; x++){
        m(x,0)= data[x*col_num + n];
    }
    return m;
}


matrix matrix::hadamard(const matrix & m) const {

    if (row_num != m.rows() || col_num!= m.cols()){
        throw Exception("matrix error: dimension mismatch with hadamard");
    }

    matrix y(row_num,col_num);

    for (int k = 0; k < m.size(); k++){
         y.data[k] = data[k]*m.data[k];
    }

    return y;
}


double matrix::dot(const matrix & m) const {

    if (row_num != m.rows() || col_num!= m.cols()){
        throw Exception("matrix error: dimension mismatch in dot");
    }

    double sum = 0;

    for (int k = 0; k < m.size(); k++){
        sum += data[k]*m.data[k];
    }

    return sum;
}


void matrix::print() const {

    for (int k = 0; k < row_num; k++){
        for (int l = 0; l < col_num; l++){
            std::cout << data[k*col_num + l] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}

double matrix::norm() const {
    double N = 0;
    for (auto it = data.begin(); it != data.end(); it++){
        N += std::pow(*it,2);
    }
    return std::sqrt(N);
}


matrix operator+(const matrix& m1, const matrix& m2){
    matrix _m(m1);
    _m += m2;
    return _m;
}



matrix operator-(const matrix& m1, const matrix& m2){
    matrix _m(m1);
    _m += (-1)*m2;
    return _m;
}



matrix operator*(double t, const matrix& m){
    matrix _m(m);
    _m *= t;
    return _m;
}



matrix operator*(const matrix& m, double t){
    return t*m;
}



matrix operator/(const matrix& m, double t){
    return (1/t)*m;
}


// matrix multiplication

matrix operator*(const matrix& m1, const matrix& m2){

    if (m1.cols() != m2.rows()){
        throw Exception("matrix error: matrix multiplication. must have m1.cols = m2.rows ");
    }

    int N = m1.cols();

    matrix m(m1.rows(),m2.cols());

    for (int x = 0; x < m.rows(); x++){
        for (int y = 0; y < m.cols(); y++){
            double elem = 0;
            for (int z = 0; z < N; z++){
                elem += m1(x,z)*m2(z,y);
            }
            m(x,y) = elem;
        }
    }

    return m;
}




vector::vector(int rows): matrix(rows,1){};


vector::vector(const matrix &m){

    if (m.cols() != 1){
        throw Exception("vector error: can't matrix array to vector since column number != 1");
    }

    row_num = m.rows();
    col_num = 1;
    this -> data = m.return_data();
}



vector::vector(const tensor & t){

    data = t.return_data();
    row_num = static_cast<int>(data.size());
    col_num = 1;
}


matrix outer_product(vector a, vector b){

    int rows = a.size();
    int cols = b.size();

    matrix M(rows,cols);

    for (int m = 0; m < rows; m++){
        double c = a(m);
        for (int n = 0; n < cols; n++){
            M(m,n) = c*b(n);
        }
    }

    return M;
}
