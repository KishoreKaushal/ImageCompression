#include<bits/stdc++.h>
using namespace std;

const int N = 8;

// int I[N][N] = {
//                 { 140 , 144 , 147 , 140 , 140 , 155 , 179 , 175},
//                 { 144 , 152 , 140 , 147 , 140 , 148 , 167 , 179},
//                 { 152 , 155 , 136 , 167 , 163 , 162 , 152 , 172},
//                 { 168 , 145 , 156 , 160 , 152 , 155 , 136 , 160},
//                 { 162 , 148 , 156 , 148 , 140 , 136 , 147 , 162},
//                 { 147 , 167 , 140 , 155 , 155 , 140 , 136 , 162},
//                 { 136 , 156 , 123 , 167 , 162 , 144 , 140 , 147},
//                 { 148 , 155 , 136 , 155 , 152 , 147 , 147 , 136}
//             };


int I[N][N] = {
    {88, 84, 83, 84, 85, 86, 83, 82},
    {86, 82, 82, 83, 82, 83, 83, 81}, 
    {82, 82, 84, 87, 87, 87, 81, 84}, 
    {81, 86, 87, 89, 82, 82, 84, 87}, 
    {81, 84, 83, 87, 85, 89, 80, 81}, 
    {81, 85, 85, 86, 81, 89, 81, 85}, 
    {82, 81, 86, 83, 86, 89, 81, 84}, 
    {88, 88, 90 ,84, 85, 88, 88, 81}
};

int DCT[N][N];
double Cosine[N][N];

double C(int x) {
    if (x != 0) return 1;
    return 1.0/sqrt(2);
}

double calc_cosine(int x, int i){
    return cos( ((2*x + 1) * i * M_PI) / (2*N) );
}

int main() {

    for(int x = 0; x < N; x++){
        for (int i = 0; i < N; i++){
            Cosine[x][i] = calc_cosine(x, i);
        }
    }
    
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            double sum = 0.0;
            for(int x = 0; x < N; x++){
                for(int y = 0; y < N; y++){
                    sum += I[x][y] * Cosine[x][i] * Cosine[y][j];
                }
            }
            // cout << sum << " ";
            sum *= (1/sqrt(2*N)) * C(i) * C(j); 
            DCT[i][j] = sum;
            cout << DCT[i][j] << " ";
        }
        cout << endl;
    }

    double sum = 0.0;
    for (int i = 0 ; i < N; i++) {
        for (int j = 0; j < N; j++) {
            sum += I[i][j];
        }
    }
    cout << "Average: " << sum / (N*N) << endl;
    return 0;
}